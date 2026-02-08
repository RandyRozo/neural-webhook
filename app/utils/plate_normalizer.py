"""
Modulo de normalizacion de placas para camaras Neural.
Implementa validacion de formato colombiano, correccion OCR y filtrado.
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Codigos ISO 3166-1 numerico de Colombia (soportar variantes)
COLOMBIA_COUNTRY_CODES = ["170", "210", ""]

# Correcciones OCR bidireccionales (letra <-> numero)
OCR_CORRECTIONS = {
    'O': '0', '0': 'O',
    'I': '1', '1': 'I',
    'Z': '2', '2': 'Z',
    'S': '5', '5': 'S',
    'B': '8', '8': 'B',
    'G': '6', '6': 'G',
}


@dataclass
class PlateNormalizationResult:
    """Resultado de la normalizacion de placa."""
    normalized_plate: Optional[str]
    vehicle_type: Optional[str]
    is_valid: bool
    is_colombian: bool
    confidence_normalized: float
    ocr_corrections_applied: int
    rejection_reason: Optional[str]


def _is_valid_colombian_plate(plate: str) -> bool:
    """
    Valida si una placa coincide con algun formato colombiano.

    Formatos validos:
    - Carros: ABC123 (3 letras + 3 numeros)
    - Motos nuevas: ABC12D (3 letras + 2 numeros + 1 letra)
    - Motos antiguas: ABC12 (3 letras + 2 numeros)
    - Motocarros: 123ABC (3 numeros + 3 letras)
    """
    if not plate:
        return False

    plate = plate.upper().strip()

    # Carros: ABC123
    if re.match(r'^[A-Z]{3}[0-9]{3}$', plate):
        return True

    # Motos nuevas: ABC12D
    if re.match(r'^[A-Z]{3}[0-9]{2}[A-Z]$', plate):
        return True

    # Motos antiguas: ABC12
    if re.match(r'^[A-Z]{3}[0-9]{2}$', plate):
        return True

    # Motocarros: 123ABC
    if re.match(r'^[0-9]{3}[A-Z]{3}$', plate):
        return True

    return False


def _get_plate_format(plate: str) -> Optional[str]:
    """Determina el formato de la placa."""
    if not plate:
        return None

    plate = plate.upper().strip()

    if re.match(r'^[A-Z]{3}[0-9]{3}$', plate):
        return 'carro'
    if re.match(r'^[A-Z]{3}[0-9]{2}[A-Z]$', plate):
        return 'moto_nueva'
    if re.match(r'^[A-Z]{3}[0-9]{2}$', plate):
        return 'moto_antigua'
    if re.match(r'^[0-9]{3}[A-Z]{3}$', plate):
        return 'motocarro'

    return None


def _apply_ocr_correction(plate: str, max_corrections: int = 1) -> tuple[str, int]:
    """
    Aplica correcciones OCR bidireccionales basadas en posicion esperada.
    """
    if not plate or len(plate) < 5:
        return plate, 0

    plate = plate.upper().strip()
    corrections = 0
    corrected = list(plate)

    # Intentar formato carro/moto (comienza con letras)
    if len(plate) >= 5:
        # Posiciones 0,1,2 deben ser letras
        for i in range(min(3, len(plate))):
            if plate[i].isdigit() and plate[i] in OCR_CORRECTIONS:
                if corrections < max_corrections:
                    corrected[i] = OCR_CORRECTIONS[plate[i]]
                    corrections += 1

        # Posiciones 3,4 (y 5 si es carro) deben ser numeros
        if len(plate) >= 5:
            for i in range(3, min(6, len(plate))):
                if i == 5 and len(plate) == 6:
                    # Formato moto nueva (ABC12D), posicion 5 debe ser letra
                    if plate[i].isdigit() and plate[i] in OCR_CORRECTIONS:
                        if corrections < max_corrections:
                            corrected[i] = OCR_CORRECTIONS[plate[i]]
                            corrections += 1
                else:
                    # Posiciones 3,4 deben ser numeros
                    if plate[i].isalpha() and plate[i] in OCR_CORRECTIONS:
                        if corrections < max_corrections:
                            corrected[i] = OCR_CORRECTIONS[plate[i]]
                            corrections += 1

    corrected_plate = ''.join(corrected)

    if _is_valid_colombian_plate(corrected_plate):
        return corrected_plate, corrections

    # Intentar formato motocarro (comienza con numeros)
    if len(plate) == 6:
        corrections = 0
        corrected = list(plate)

        # Posiciones 0,1,2 deben ser numeros
        for i in range(3):
            if plate[i].isalpha() and plate[i] in OCR_CORRECTIONS:
                if corrections < max_corrections:
                    corrected[i] = OCR_CORRECTIONS[plate[i]]
                    corrections += 1

        # Posiciones 3,4,5 deben ser letras
        for i in range(3, 6):
            if plate[i].isdigit() and plate[i] in OCR_CORRECTIONS:
                if corrections < max_corrections:
                    corrected[i] = OCR_CORRECTIONS[plate[i]]
                    corrections += 1

        corrected_plate = ''.join(corrected)
        if _is_valid_colombian_plate(corrected_plate):
            return corrected_plate, corrections

    return plate, 0


def normalize_neural_plate(
    plate_text: Optional[str],
    confidence: float,
    country: Optional[str],
    vehicle_type: Optional[str],
    config
) -> PlateNormalizationResult:
    """
    Normaliza y valida una placa detectada por camara Neural.

    PRIORIDAD:
    1. Validar por FORMATO de placa colombiana (independiente del codigo de pais)
    2. Si el formato es valido -> ACEPTAR
    3. Solo si el formato no coincide -> verificar codigo de pais
    """
    # Normalizar confianza
    confidence_normalized = confidence / 100.0 if confidence > 1.0 else confidence
    confidence_normalized = max(0.0, min(1.0, confidence_normalized))

    # Configuracion
    min_confidence = getattr(config, 'min_confidence_neural', 85.0) / 100.0
    reject_foreign = getattr(config, 'reject_foreign_plates', True)
    max_corrections = getattr(config, 'max_ocr_corrections_neural', 1)

    # Validar que hay placa
    if not plate_text or plate_text.lower() in ['unknown', 'none', '']:
        return PlateNormalizationResult(
            normalized_plate=None,
            vehicle_type=vehicle_type,
            is_valid=False,
            is_colombian=False,
            confidence_normalized=confidence_normalized,
            ocr_corrections_applied=0,
            rejection_reason="Placa vacia o desconocida"
        )

    # Limpiar texto de placa
    plate_clean = plate_text.upper().strip().replace(' ', '').replace('-', '')

    # Validar confianza minima
    if confidence_normalized < min_confidence:
        logger.info(f"Placa rechazada por baja confianza: {plate_clean} ({confidence_normalized:.2%} < {min_confidence:.2%})")
        return PlateNormalizationResult(
            normalized_plate=plate_clean,
            vehicle_type=vehicle_type,
            is_valid=False,
            is_colombian=False,
            confidence_normalized=confidence_normalized,
            ocr_corrections_applied=0,
            rejection_reason=f"Confianza insuficiente ({confidence_normalized:.2%} < {min_confidence:.2%})"
        )

    # Validar PRIMERO por formato, DESPUES por codigo de pais
    is_format_colombian = _is_valid_colombian_plate(plate_clean)
    is_country_colombian = country in COLOMBIA_COUNTRY_CODES or country is None

    # Si el formato es valido colombiano -> ACEPTAR (ignorar codigo de pais incorrecto)
    if is_format_colombian:
        plate_format = _get_plate_format(plate_clean)
        logger.info(f"Placa valida por formato (formato {plate_format}): {plate_clean} (pais reportado: {country})")
        return PlateNormalizationResult(
            normalized_plate=plate_clean,
            vehicle_type=vehicle_type,
            is_valid=True,
            is_colombian=True,
            confidence_normalized=confidence_normalized,
            ocr_corrections_applied=0,
            rejection_reason=None
        )

    # Intentar correccion OCR
    corrected_plate, corrections = _apply_ocr_correction(plate_clean, max_corrections)

    if corrections > 0 and _is_valid_colombian_plate(corrected_plate):
        plate_format = _get_plate_format(corrected_plate)
        logger.info(f"Placa corregida (formato {plate_format}): {plate_clean} -> {corrected_plate} ({corrections} correcciones)")
        return PlateNormalizationResult(
            normalized_plate=corrected_plate,
            vehicle_type=vehicle_type,
            is_valid=True,
            is_colombian=True,
            confidence_normalized=confidence_normalized,
            ocr_corrections_applied=corrections,
            rejection_reason=None
        )

    # Si llegamos aqui: formato NO colombiano
    # Solo rechazar por pais extranjero si reject_foreign=True
    if reject_foreign and not is_country_colombian:
        logger.info(f"Placa extranjera rechazada: {plate_clean} (pais: {country})")
        return PlateNormalizationResult(
            normalized_plate=plate_clean,
            vehicle_type=vehicle_type,
            is_valid=False,
            is_colombian=False,
            confidence_normalized=confidence_normalized,
            ocr_corrections_applied=0,
            rejection_reason=f"Placa extranjera (codigo pais: {country})"
        )

    # Formato invalido (ni colombiano ni extranjero reconocido)
    logger.info(f"Placa rechazada por formato invalido: {plate_clean}")
    return PlateNormalizationResult(
        normalized_plate=plate_clean,
        vehicle_type=vehicle_type,
        is_valid=False,
        is_colombian=False,
        confidence_normalized=confidence_normalized,
        ocr_corrections_applied=0,
        rejection_reason=f"Formato de placa invalido: {plate_clean}"
    )
