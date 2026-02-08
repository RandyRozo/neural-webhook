import logging
import os
import sys
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

from config import NeuralConfig
from services.database_service import NeuralDatabaseService
from services.event_processor import NeuralEventProcessor
from services.vault_secret_provider import VaultSecretProvider


def setup_logging(config: NeuralConfig):
    """Configura el sistema de logging"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"

    handlers = [logging.StreamHandler()]

    if not config.is_kubernetes_environment():
        os.makedirs("logs", exist_ok=True)
        handlers.append(logging.FileHandler("logs/neural-webhook.log"))

    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format=log_format,
        handlers=handlers,
        force=True,
    )


logger = logging.getLogger(__name__)

# Variables globales
db_service: Optional[NeuralDatabaseService] = None
event_processor: Optional[NeuralEventProcessor] = None
config: Optional[NeuralConfig] = None


class NeuralWebhookService:
    """Servicio principal para webhook de camaras Neural"""

    def __init__(self):
        global config
        config = NeuralConfig()
        setup_logging(config)

        self.config = config
        self.db_service = None
        self.event_processor = None
        self.vault_provider = None
        self._health_status = {"status": "starting", "services": {}}

        if self.config.vault_enabled:
            self.vault_provider = VaultSecretProvider(
                vault_id=self.config.vault_id,
                auth_type=self.config.oracle_auth_type,
                cache_ttl=self.config.vault_cache_ttl,
            )

        logger.info(f"Inicializando Neural Webhook Service v1.0.0")
        logger.info(f"Worker ID: {self.config.worker_id}")
        logger.info(
            f"Environment: {'Kubernetes' if self.config.is_kubernetes_environment() else 'Local'}"
        )
        logger.info(f"Storage Type: {self.config.storage_type}")

        if self.config.storage_type == "oracle_cloud":
            logger.info(f"Oracle Cloud - Bucket: {self.config.oracle_bucket_name}")
            logger.info(f"Oracle Cloud - Region: {self.config.oracle_region}")

    async def _handle_db_auth_error(self, exc: Exception) -> bool:
        """Callback invocado cuando una operacion de BD falla por autenticacion.
        Refresca la contrasena desde vault y recrea los pools."""
        if not self.vault_provider:
            return False

        try:
            logger.warning(f"Manejando error de autenticacion de BD: {exc}")

            self.vault_provider.invalidate_all()

            db_password = await self.vault_provider.get_secret(
                self.config.secret_name_db_password, force_refresh=True
            )

            self.config.update_db_password(db_password)

            await self.db_service.recreate_pools()

            logger.info("Credenciales de BD refrescadas y pools recreados exitosamente")
            return True

        except Exception as refresh_e:
            logger.error(f"Error refrescando credenciales de BD: {refresh_e}")
            return False

    async def initialize_services(self):
        """Inicializa los servicios"""
        try:
            # Obtener contrasena de BD desde OCI Vault si esta habilitado
            if self.vault_provider:
                logger.info("Obteniendo DB_PASSWORD desde OCI Vault...")
                self.vault_provider.initialize()
                db_password = await self.vault_provider.get_secret(
                    self.config.secret_name_db_password
                )
                self.config.update_db_password(db_password)
                logger.info("DB_PASSWORD obtenido desde OCI Vault exitosamente")

            logger.info("Inicializando servicio de base de datos...")
            self.db_service = NeuralDatabaseService(self.config)
            await self.db_service.initialize()
            self._health_status["services"]["database"] = "healthy"

            # Registrar callback para renovacion automatica de credenciales
            if self.vault_provider:
                self.db_service.on_auth_error = self._handle_db_auth_error

            logger.info("Inicializando procesador de eventos...")
            self.event_processor = NeuralEventProcessor(self.config, self.db_service)
            self._health_status["services"]["event_processor"] = "healthy"

            logger.info("Verificando servicio de almacenamiento...")
            storage_health = await self.event_processor.storage_service.health_check()
            if storage_health.get("status") == "healthy":
                self._health_status["services"]["storage"] = "healthy"
                logger.info(f"Storage healthy: {storage_health.get('storage_type')}")
            else:
                self._health_status["services"]["storage"] = "unhealthy"
                logger.warning(f"Storage unhealthy: {storage_health}")

            global db_service, event_processor
            db_service = self.db_service
            event_processor = self.event_processor

            self._health_status["status"] = "healthy"
            logger.info("Servicios inicializados exitosamente")

        except Exception as e:
            self._health_status["status"] = "unhealthy"
            self._health_status["error"] = str(e)
            logger.error(f"Error inicializando servicios: {e}", exc_info=True)
            raise

    async def shutdown_services(self):
        """Cierra los servicios"""
        logger.info("Cerrando servicios...")

        if self.db_service:
            await self.db_service.close()
            self._health_status["services"]["database"] = "stopped"

        self._health_status["status"] = "stopped"
        logger.info("Servicios cerrados correctamente")

    def get_health_status(self) -> dict:
        """Obtiene el estado de salud del servicio"""
        return {
            **self._health_status,
            "worker_id": self.config.worker_id,
            "node_name": self.config.node_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


service = NeuralWebhookService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Maneja el ciclo de vida de la aplicacion"""
    try:
        await service.initialize_services()
        logger.info("Aplicacion Neural Webhook iniciada")
        yield
    except Exception as e:
        logger.error(f"Error durante startup: {e}", exc_info=True)
        raise
    finally:
        await service.shutdown_services()
        logger.info("Aplicacion Neural Webhook detenida")


app = FastAPI(
    title="Neural Camera Webhook",
    description="Microservicio webhook para camaras Neural con ANPR y Oracle Cloud Storage",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== ENDPOINTS ====================


@app.get("/", tags=["Health"])
async def root():
    """Endpoint raiz"""
    return {
        "service": "Neural Camera Webhook",
        "version": "1.0.0",
        "status": "active",
        "worker_id": config.worker_id if config else "unknown",
        "storage_type": config.storage_type if config else "unknown",
        "bucket": config.oracle_bucket_name if config else "unknown",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/metrics", tags=["Health"])
async def metrics():
    """Endpoint de metricas"""
    return "ok"


@app.post("/events", tags=["Webhook"])
async def receive_event(request: Request):
    """
    Recibe eventos JSON de camaras Neural
    - JSON con datos de placa, confianza e imagenes en base64
    """
    if not event_processor:
        logger.error("Event processor no disponible")
        raise HTTPException(status_code=503, detail="Servicio no disponible")

    try:
        logger.info("REQUEST RECIBIDO en /events")
        event_id, response_data = await event_processor.process_webhook_event(request)

        logger.info(
            f"Evento procesado: event_id={event_id}, status={response_data.get('status')}, "
            f"created={response_data.get('events_created', 0)}, rejected={response_data.get('events_rejected', 0)}"
        )

        # Siempre retornar HTTP 200 si el webhook proceso el request correctamente
        # La camara debe recibir 200 OK independientemente de si se acepto o rechazo la placa
        status_code = 200 if response_data.get("status") == "ok" else 500

        return JSONResponse(content=response_data, status_code=status_code)

    except Exception as e:
        logger.error(f"ERROR en webhook endpoint: {e}", exc_info=True)
        logger.error(f"Traceback completo:\n{traceback.format_exc()}")
        return JSONResponse(
            content={
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            status_code=500,
        )


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check para Kubernetes"""
    health_status = service.get_health_status()

    if health_status["status"] not in ["healthy"]:
        raise HTTPException(status_code=503, detail="Servicio no saludable")

    storage_health = None
    if event_processor:
        try:
            processor_health = await event_processor.health_check()
            storage_health = processor_health.get("storage")
        except Exception as e:
            logger.warning(f"Error checking processor health: {e}")

    response_data = {
        "status": health_status["status"],
        "worker_id": health_status["worker_id"],
        "node_name": health_status["node_name"],
        "services": health_status["services"],
        "total_events": event_processor.total_events_processed
        if event_processor
        else 0,
        "timestamp": health_status["timestamp"],
    }

    if storage_health:
        response_data["storage"] = storage_health

    return response_data


@app.get("/ready", tags=["Health"])
async def readiness_check():
    """Readiness check para Kubernetes"""
    if not event_processor or not db_service:
        raise HTTPException(status_code=503, detail="Servicios no inicializados")

    storage_ready = False
    try:
        processor_health = await event_processor.health_check()
        storage_ready = processor_health.get("storage", {}).get("status") == "healthy"
    except Exception as e:
        logger.warning(f"Storage readiness check failed: {e}")

    if not storage_ready:
        raise HTTPException(status_code=503, detail="Storage no disponible")

    return {
        "status": "ready",
        "storage_type": config.storage_type if config else "unknown",
        "bucket": config.oracle_bucket_name if config else "unknown",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/eventos/recientes", tags=["Events"])
async def get_eventos_recientes(limit: int = 10):
    """Obtiene eventos recientes"""
    if not db_service:
        raise HTTPException(status_code=503, detail="Servicio no disponible")

    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="Limite debe estar entre 1 y 100")

    try:
        eventos = await db_service.get_recent_events(limit)
        return {
            "eventos": eventos,
            "total": len(eventos),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Error obteniendo eventos: {e}")
        raise HTTPException(status_code=500, detail="Error interno")


@app.get("/eventos/placa/{plate}", tags=["Events"])
async def get_eventos_por_placa(plate: str, limit: int = 10):
    """Obtiene eventos por placa"""
    if not db_service:
        raise HTTPException(status_code=503, detail="Servicio no disponible")

    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="Limite debe estar entre 1 y 100")

    try:
        eventos = await db_service.get_events_by_plate(plate, limit)
        return {
            "plate": plate,
            "eventos": eventos,
            "total": len(eventos),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Error obteniendo eventos por placa: {e}")
        raise HTTPException(status_code=500, detail="Error interno")


@app.get("/estadisticas", tags=["Stats"])
async def get_estadisticas():
    """Obtiene estadisticas de eventos"""
    if not db_service:
        raise HTTPException(status_code=503, detail="Servicio no disponible")

    try:
        stats = await db_service.get_event_stats()

        runtime_stats = {
            "session_events": event_processor.total_events_processed
            if event_processor
            else 0,
            "last_event_time": event_processor.last_event_time.isoformat()
            if event_processor and event_processor.last_event_time
            else None,
            "worker_id": config.worker_id if config else "unknown",
            "storage_type": config.storage_type if config else "unknown",
            "bucket": config.oracle_bucket_name if config else "unknown",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return {**stats, **runtime_stats}
    except Exception as e:
        logger.error(f"Error obteniendo estadisticas: {e}")
        raise HTTPException(status_code=500, detail="Error interno")


@app.get("/imagen/{image_path:path}", tags=["Images"])
async def get_image_url(image_path: str, expires_in: int = 3600):
    """Obtiene URL pre-firmada para una imagen"""
    if not event_processor:
        raise HTTPException(status_code=503, detail="Servicio no disponible")

    if expires_in < 60 or expires_in > 86400:
        raise HTTPException(
            status_code=400, detail="expires_in debe estar entre 60 y 86400"
        )

    try:
        url = await event_processor.storage_service.get_image_url(
            image_path, expires_in
        )
        if not url:
            raise HTTPException(status_code=404, detail="Imagen no encontrada")

        return {
            "image_path": image_path,
            "url": url,
            "expires_in": expires_in,
            "storage_type": config.storage_type if config else "unknown",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error obteniendo URL: {e}")
        raise HTTPException(status_code=500, detail="Error interno")


@app.get("/configuracion", tags=["Config"])
async def get_configuracion():
    """Informacion de configuracion"""
    if not config:
        raise HTTPException(status_code=503, detail="Configuracion no disponible")

    return {
        "database": config.get_database_info(),
        "storage": config.get_storage_info(),
        "worker_id": config.worker_id,
        "node_name": config.node_name,
        "kubernetes": config.is_kubernetes_environment(),
        "log_level": config.log_level,
        "version": "1.0.0",
    }


@app.get("/storage/test", tags=["Storage"])
async def test_storage():
    """Test de almacenamiento"""
    if not event_processor:
        raise HTTPException(status_code=503, detail="Servicio no disponible")

    try:
        test_image = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00\xaa\xff\xd9"

        (
            relative_path,
            full_bucket_url,
        ) = await event_processor.storage_service.save_image(test_image, "TEST")
        image_url = await event_processor.storage_service.get_image_url(
            relative_path, 300
        )
        health = await event_processor.storage_service.health_check()
        deleted = await event_processor.storage_service.delete_image(relative_path)

        return {
            "status": "success",
            "storage_type": config.storage_type if config else "unknown",
            "bucket": config.oracle_bucket_name if config else "unknown",
            "test_image_path": relative_path,
            "test_image_url": full_bucket_url,
            "test_presigned_url": image_url,
            "test_deleted": deleted,
            "health_check": health,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Error en test de almacenamiento: {e}")
        return {
            "status": "error",
            "error": str(e),
            "storage_type": config.storage_type if config else "unknown",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


def main():
    """Funcion principal"""
    try:
        uvicorn_config = {
            "app": "main:app",
            "host": "0.0.0.0",
            "port": 8000,
            "log_level": "info",
            "access_log": True,
            "loop": "asyncio",
        }

        if not os.environ.get("PRODUCTION", False):
            uvicorn_config["reload"] = True

        logger.info("Iniciando servidor Neural Webhook en puerto 8000...")
        uvicorn.run(**uvicorn_config)

    except KeyboardInterrupt:
        logger.info("Servidor detenido por el usuario")
    except Exception as e:
        logger.error(f"Error fatal: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
