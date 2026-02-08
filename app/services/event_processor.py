import base64
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from fastapi import Request

from services.storage_service import NeuralStorageService
from utils.plate_normalizer import normalize_neural_plate

logger = logging.getLogger(__name__)


class NeuralEventProcessor:
    """Procesador de eventos para camaras Neural"""

    def __init__(self, config, db_service):
        self.config = config
        self.db_service = db_service
        self.storage_service = NeuralStorageService(config)
        self._total_events = 0
        self._last_event_time: Optional[datetime] = None

    @property
    def total_events_processed(self) -> int:
        return self._total_events

    @property
    def last_event_time(self) -> Optional[datetime]:
        return self._last_event_time

    async def process_webhook_event(self, request: Request) -> Tuple[Optional[int], Dict[str, Any]]:
        try:
            logger.info("EVENTO NEURAL")
            logger.info("=" * 50)

            # Leer body JSON
            body_bytes = await request.body()
            data = json.loads(body_bytes.decode("utf-8"))

            # Extraer datos de infoplate
            infoplate = data.get("infoplate", {})

            plate = infoplate.get("Plate", "")
            timestamp = infoplate.get("DateHour", datetime.now(timezone.utc).isoformat())
            confidence_str = infoplate.get("confidence", "0")
            camera_name = infoplate.get("CamName", "")
            image_b64 = infoplate.get("img", "")
            evidences = infoplate.get("Evidences", [])

            # Normalizar confianza
            try:
                confidence = float(confidence_str)
            except (ValueError, TypeError):
                confidence = 0.0

            if plate == "UNKNOWN" or not plate:
                plate = None

            logger.info(f"  Placa: {plate if plate else 'No detectada'}")
            logger.info(f"  Confianza: {confidence}%")
            logger.info(f"  Timestamp: {timestamp}")
            logger.info(f"  Camara: {camera_name}")
            logger.info(f"  Imagen principal: {'Si' if image_b64 else 'No'}")
            logger.info(f"  Evidencias: {len(evidences)}")

            # Variables para respuesta
            created_events = []
            rejected_events = []

            # NORMALIZACION de placa
            confidence_for_normalize = confidence

            logger.info(f"Normalizando: plate={plate}, conf={confidence_for_normalize}")

            result = normalize_neural_plate(
                plate_text=plate,
                confidence=confidence_for_normalize,
                country=None,  # Neural no envia codigo de pais
                vehicle_type=None,  # Neural no envia tipo de vehiculo
                config=self.config,
            )

            logger.info(
                f"Resultado: valid={result.is_valid}, colombian={result.is_colombian}, "
                f"normalized={result.normalized_plate}, ocr={result.ocr_corrections_applied}"
            )

            # RECHAZADA
            if not result.is_valid:
                logger.warning(f"RECHAZADA: {result.rejection_reason}")

                if not self.config.strict_mode:
                    try:
                        rejection_id = await self.db_service.save_rejected_plate_wh(
                            {
                                "camera_brand": "neural",
                                "camera_id": camera_name,
                                "raw_plate_text": plate,
                                "confidence": result.confidence_normalized,
                                "rejection_reason": result.rejection_reason,
                                "rejection_type": self._classify_rejection_type(result),
                                "country": None,
                                "vehicle_type": None,
                                "raw_data": json.dumps(
                                    {k: v for k, v in infoplate.items() if k not in ("img", "Evidences")},
                                    ensure_ascii=False,
                                ),
                            }
                        )
                        logger.info(f"Rechazo guardado ID: {rejection_id}")
                    except Exception as e:
                        logger.error(f"Error guardando rechazo: {e}")

                rejected_events.append(
                    {
                        "raw_plate": plate,
                        "confidence": result.confidence_normalized,
                        "rejection_reason": result.rejection_reason,
                    }
                )

            # VALIDA - Guardar imagenes y evento
            else:
                logger.info("VALIDA - Guardando...")

                saved_images = []

                # Guardar imagen principal
                if image_b64:
                    try:
                        img_bytes = self._decode_base64_image(image_b64)
                        if img_bytes:
                            filename = f"detection_{datetime.now(timezone.utc).strftime('%H%M%S_%f')}.jpg"
                            relative_path, full_bucket_url = await self.storage_service.save_image(
                                img_bytes,
                                filename,
                                plate_prefix=result.normalized_plate if result.normalized_plate else "unknown",
                            )
                            saved_images.append(full_bucket_url)
                            logger.info(f"Imagen principal guardada: {relative_path}")
                    except Exception as e:
                        logger.error(f"Error guardando imagen principal: {e}")

                # Guardar evidencias
                if evidences:
                    for idx, evidence_item in enumerate(evidences, start=1):
                        try:
                            evidence_data = evidence_item.get("Evidence", {})
                            evidence_b64 = evidence_data.get("imgEV", "")
                            if evidence_b64:
                                img_bytes = self._decode_base64_image(evidence_b64)
                                if img_bytes:
                                    filename = f"evidence_{idx}_{datetime.now(timezone.utc).strftime('%H%M%S_%f')}.jpg"
                                    relative_path, full_bucket_url = await self.storage_service.save_image(
                                        img_bytes,
                                        filename,
                                        plate_prefix=result.normalized_plate if result.normalized_plate else "unknown",
                                    )
                                    saved_images.append(full_bucket_url)
                                    logger.info(f"Evidencia {idx} guardada: {relative_path}")
                        except Exception as e:
                            logger.error(f"Error guardando evidencia {idx}: {e}")

                # Guardar evento en BD
                capture_time = self._parse_neural_timestamp(timestamp)
                ocr_report = self._generate_ocr_correction_report(
                    plate, result.normalized_plate, result.ocr_corrections_applied
                )

                # Usar la primera imagen como URL principal
                primary_image_url = saved_images[0] if saved_images else None

                event_record = {
                    "plate": result.normalized_plate,
                    "confidence": result.confidence_normalized,
                    "capture_time": capture_time,
                    "camera_id": camera_name,
                    "camera_brand": "neural",
                    "camera_location": None,
                    "image_url": primary_image_url,
                    "vehicle_type": result.vehicle_type,
                    "direction": None,
                    "violation_type": None,
                    "raw_data": json.dumps(
                        {k: v for k, v in infoplate.items() if k not in ("img", "Evidences")},
                        ensure_ascii=False,
                    ),
                    "ocr_correction_report": ocr_report,
                }

                event_id = await self.db_service.save_event(event_record)

                created_events.append(
                    {
                        "event_id": event_id,
                        "image_urls": saved_images,
                        "normalized_plate": result.normalized_plate,
                        "ocr_corrections": result.ocr_corrections_applied,
                    }
                )

                logger.info(f"Evento ID: {event_id}, Placa: {result.normalized_plate}")

            logger.info(f"\nRESUMEN:")
            logger.info(f"  - Creados: {len(created_events)}")
            logger.info(f"  - Rechazados: {len(rejected_events)}")

            self._total_events += len(created_events)
            self._last_event_time = datetime.now(timezone.utc)

            response_data = {
                "status": "ok",
                "events_created": len(created_events),
                "events_rejected": len(rejected_events),
                "plate": plate if plate else "No detectada",
                "confidence": confidence_str,
                "capture_time": timestamp,
                "camera_name": camera_name,
                "total_events": self._total_events,
                "timestamp": self._last_event_time.isoformat(),
            }

            if created_events:
                response_data["events"] = created_events
            if rejected_events:
                response_data["rejections"] = rejected_events

            first_event_id = created_events[0]["event_id"] if created_events else None

            logger.info("Completado")
            logger.info("=" * 50)
            return first_event_id, response_data

        except Exception as e:
            logger.error(f"ERROR CRITICO: {e}", exc_info=True)
            return None, {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    def _decode_base64_image(self, b64_string: str) -> Optional[bytes]:
        """Decodifica una imagen base64, limpiando prefijos si existen."""
        try:
            if not b64_string:
                return None

            # Limpiar prefijo data:image/...;base64,
            if "," in b64_string:
                b64_string = b64_string.split(",")[1]

            return base64.b64decode(b64_string)
        except Exception as e:
            logger.error(f"Error decodificando base64: {e}")
            return None

    def _parse_neural_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Parsea el timestamp de Neural."""
        try:
            if not timestamp_str:
                return None
            # Intentar formato ISO
            return datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except Exception:
            try:
                # Intentar otros formatos comunes
                for fmt in ["%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%Y%m%d%H%M%S"]:
                    try:
                        return datetime.strptime(timestamp_str, fmt).replace(
                            tzinfo=timezone.utc
                        )
                    except ValueError:
                        continue
            except Exception:
                pass
            return None

    def _classify_rejection_type(self, result) -> str:
        """Clasifica el tipo de rechazo."""
        if result.confidence_normalized < (self.config.min_confidence_neural / 100.0):
            return "confidence"
        if not result.is_colombian:
            return "foreign_plate"
        return "format"

    def _generate_ocr_correction_report(
        self,
        original_plate: Optional[str],
        normalized_plate: Optional[str],
        corrections_count: int,
    ) -> str:
        """Genera reporte de correcciones OCR."""
        if corrections_count == 0 or not original_plate or not normalized_plate:
            return "Sin correcciones aplicadas"
        if original_plate == normalized_plate:
            return "Sin correcciones aplicadas"

        original_clean = original_plate.upper().strip().replace(" ", "").replace("-", "")
        normalized_clean = (
            normalized_plate.upper().strip().replace(" ", "").replace("-", "")
        )

        corrections_details = []
        min_len = min(len(original_clean), len(normalized_clean))

        for i in range(min_len):
            if original_clean[i] != normalized_clean[i]:
                corrections_details.append(
                    f"pos{i}: {original_clean[i]}->{normalized_clean[i]}"
                )

        if corrections_details:
            details_str = ", ".join(corrections_details)
            return f"{original_clean} -> {normalized_clean} ({corrections_count} corr: {details_str})"
        return "Sin correcciones aplicadas"

    async def health_check(self) -> Dict[str, Any]:
        """Health check del procesador."""
        storage_health = await self.storage_service.health_check()
        return {
            "processor": "healthy",
            "total_events": self._total_events,
            "last_event_time": self._last_event_time.isoformat()
            if self._last_event_time
            else None,
            "storage": storage_health,
        }
