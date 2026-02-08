import logging
import os
import sys
from pathlib import Path
from typing import Optional


class NeuralConfig:
    """
    Configuracion centralizada para microservicio Neural Webhook
    Compatible con despliegue en Kubernetes y Oracle Cloud Object Storage
    """

    def __init__(self):
        # Configuracion de Base de Datos - Endpoints separados de lectura y escritura
        self.db_name = self._get_env("DB_NAME")
        self.db_user = self._get_env("DB_USER")

        # OCI Vault para DB_PASSWORD (opcional - fallback a env var)
        self.vault_id = os.environ.get("VAULT_ID", "")
        self.secret_name_db_password = os.environ.get("SECRET_DB_PASSWORD", "")
        self.vault_cache_ttl = int(os.environ.get("VAULT_CACHE_TTL_SEC", "86400"))
        self.vault_enabled = bool(self.vault_id and self.secret_name_db_password)

        if self.vault_enabled:
            # Se poblara desde OCI Vault antes de inicializar la BD
            self.db_password = ""
            logging.info("Vault habilitado - DB_PASSWORD se obtendra de OCI Vault")
        else:
            self.db_password = self._get_env("DB_PASSWORD")

        # Endpoint de lectura (para consultas)
        self.db_read_host = os.environ.get(
            "DB_READ_HOST", self._get_env("DB_WRITE_HOST")
        )
        self.db_read_port = int(
            os.environ.get("DB_READ_PORT", os.environ.get("DB_WRITE_PORT", "5432"))
        )

        # Endpoint de escritura (para inserts/updates)
        self.db_write_host = self._get_env("DB_WRITE_HOST")
        self.db_write_port = int(os.environ.get("DB_WRITE_PORT", "5432"))

        # Configuracion del pool de conexiones
        self.db_min_connections = int(os.environ.get("DB_MIN_CONNECTIONS", "5"))
        self.db_max_connections = int(os.environ.get("DB_MAX_CONNECTIONS", "15"))
        self.db_query_timeout = int(os.environ.get("DB_QUERY_TIMEOUT", "10"))
        self.connection_timeout = int(os.environ.get("CONNECTION_TIMEOUT", "3"))

        # Configuracion de Oracle Cloud Object Storage
        self.storage_type = os.environ.get(
            "STORAGE_TYPE", "local"
        )  # 'local' o 'oracle_cloud'
        self.oracle_namespace = os.environ.get("ORACLE_NAMESPACE", "")
        self.oracle_bucket_name = os.environ.get(
            "ORACLE_BUCKET_NAME", "webhook_cameras_prod"
        )
        self.oracle_region = os.environ.get("ORACLE_REGION", "us-ashburn-1")
        self.oracle_auth_type = os.environ.get("ORACLE_AUTH_TYPE", "instance_principal")

        # Configuracion de Normalizacion de Placas
        self.min_confidence_neural = float(
            os.environ.get("MIN_CONFIDENCE_NEURAL", "85.0")
        )
        self.reject_foreign_plates = (
            os.environ.get("REJECT_FOREIGN_PLATES", "true").lower() == "true"
        )
        self.max_ocr_corrections_neural = int(
            os.environ.get("MAX_OCR_CORRECTIONS_NEURAL", "1")
        )

        # Modo estricto: si es true, solo logs de rechazos; si es false, guarda rechazos en tabla de auditoria
        self.strict_mode = os.environ.get("STRICT_MODE", "false").lower() == "true"

        # Configuracion de Evidencias
        self.evidence_folder = os.environ.get("EVIDENCE_FOLDER", "evidencias_neural")

        # Configuracion de Workers
        self.worker_id = os.environ.get("HOSTNAME", f"neural-webhook-{os.getpid()}")
        self.node_name = os.environ.get("NODE_NAME", "unknown")

        # Configuracion de Health Checks
        self.health_check_interval = int(os.environ.get("HEALTH_CHECK_INTERVAL", "30"))
        self.health_check_timeout = int(os.environ.get("HEALTH_CHECK_TIMEOUT", "5"))

        # Configuracion de Logging
        self.log_level = os.environ.get("LOG_LEVEL", "INFO").upper()

        # URLs de base de datos derivadas
        self.database_write_url = self._build_database_url(
            self.db_write_host, self.db_write_port
        )
        self.database_read_url = self._build_database_url(
            self.db_read_host, self.db_read_port
        )

        # Crear directorios necesarios (solo si es almacenamiento local)
        if self.storage_type == "local":
            self._ensure_directories()

        # Validar configuracion critica
        self._validate_config()

    def _get_env(self, key: str) -> str:
        """Obtiene variable de entorno con manejo de errores"""
        value = os.environ.get(key)
        if not value:
            logging.error(f"Variable de entorno requerida faltante: {key}")
            sys.exit(1)
        return value

    def _build_database_url(self, host: str, port: int) -> str:
        """Construye URL de conexion PostgreSQL para asyncpg"""
        return f"postgresql+asyncpg://{self.db_user}:{self.db_password}@{host}:{port}/{self.db_name}"

    def _ensure_directories(self):
        """Crear directorios necesarios (solo para almacenamiento local)"""
        try:
            evidence_path = Path(self.evidence_folder)
            evidence_path.mkdir(exist_ok=True, parents=True)

            logs_path = Path("logs")
            logs_path.mkdir(exist_ok=True, parents=True)

        except Exception as e:
            logging.warning(f"No se pudieron crear directorios: {e}")

    def _validate_config(self):
        """Validar configuracion critica"""
        errors = []

        if self.vault_enabled:
            if not all([self.db_name, self.db_user, self.db_write_host]):
                errors.append("Configuracion de base de datos incompleta (vault mode)")
        else:
            if not all(
                [self.db_name, self.db_user, self.db_password, self.db_write_host]
            ):
                errors.append("Configuracion de base de datos incompleta")

        if self.storage_type == "oracle_cloud":
            if not all([self.oracle_namespace, self.oracle_bucket_name]):
                errors.append("Configuracion de Oracle Cloud Object Storage incompleta")

        if not (1 <= self.db_min_connections <= self.db_max_connections <= 50):
            errors.append("Configuracion de pool de conexiones invalida")

        if self.storage_type not in ["local", "oracle_cloud"]:
            errors.append("STORAGE_TYPE debe ser 'local' o 'oracle_cloud'")

        if errors:
            for error in errors:
                logging.error(error)
            sys.exit(1)

        logging.info("Validacion de configuracion completada exitosamente")

    def update_db_password(self, db_password: str) -> None:
        """Actualiza DB_PASSWORD y reconstruye las URLs de conexion.
        Usado tras obtener la contrasena desde OCI Vault."""
        self.db_password = db_password
        self.database_write_url = self._build_database_url(
            self.db_write_host, self.db_write_port
        )
        self.database_read_url = self._build_database_url(
            self.db_read_host, self.db_read_port
        )
        logging.info("DB password y URLs de conexion actualizados")

    def get_database_info(self) -> dict:
        """Obtiene informacion de configuracion de base de datos"""
        return {
            "write_endpoint": f"{self.db_write_host}:{self.db_write_port}",
            "read_endpoint": f"{self.db_read_host}:{self.db_read_port}",
            "database": self.db_name,
            "pool_config": {
                "min_connections": self.db_min_connections,
                "max_connections": self.db_max_connections,
                "query_timeout": self.db_query_timeout,
                "connection_timeout": self.connection_timeout,
            },
        }

    def get_storage_info(self) -> dict:
        """Obtiene informacion de configuracion de almacenamiento"""
        storage_info = {
            "type": self.storage_type,
            "evidence_folder": self.evidence_folder,
        }

        if self.storage_type == "oracle_cloud":
            storage_info.update(
                {
                    "namespace": self.oracle_namespace,
                    "bucket": self.oracle_bucket_name,
                    "region": self.oracle_region,
                    "auth_type": self.oracle_auth_type,
                }
            )

        return storage_info

    def get_plate_normalization_info(self) -> dict:
        """Obtiene informacion de configuracion de normalizacion de placas"""
        return {
            "min_confidence_neural": self.min_confidence_neural,
            "reject_foreign_plates": self.reject_foreign_plates,
            "max_ocr_corrections_neural": self.max_ocr_corrections_neural,
            "strict_mode": self.strict_mode,
        }

    def get_oracle_endpoint(self) -> str:
        """Obtiene el endpoint nativo de Oracle Cloud Object Storage"""
        return f"https://objectstorage.{self.oracle_region}.oraclecloud.com"

    def is_kubernetes_environment(self) -> bool:
        """Detecta si esta ejecutandose en Kubernetes"""
        return (
            os.path.exists("/var/run/secrets/kubernetes.io/serviceaccount")
            or os.environ.get("KUBERNETES_SERVICE_HOST") is not None
        )

    def __repr__(self):
        return f"<NeuralConfig(worker_id='{self.worker_id}', storage='{self.storage_type}', k8s={self.is_kubernetes_environment()})>"
