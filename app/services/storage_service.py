import os
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Tuple
import aiofiles
import oci
from oci.object_storage import ObjectStorageClient
from oci.exceptions import ServiceError

logger = logging.getLogger(__name__)


class NeuralStorageService:
    """
    Servicio de almacenamiento para Neural usando Oracle Cloud Object Storage
    Comparte el bucket con otros servicios de webhook
    """

    def __init__(self, config):
        self.config = config
        self.storage_type = config.storage_type

        if self.storage_type == "local":
            self._setup_local_storage()
        elif self.storage_type == "oracle_cloud":
            self._setup_oracle_cloud_storage()

    def _setup_local_storage(self):
        """Configurar almacenamiento local"""
        self.base_folder = Path(f"/app/{self.config.evidence_folder}")
        self.base_folder.mkdir(exist_ok=True, parents=True)
        logger.info(f"Local storage configured: {self.base_folder}")

    def _setup_oracle_cloud_storage(self):
        """Configurar Oracle Cloud Object Storage con Instance Principal"""
        try:
            if self.config.oracle_auth_type == "instance_principal":
                logger.info("Configurando autenticacion con Instance Principal...")
                signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
            else:
                logger.info("Configurando autenticacion con archivo de configuracion...")
                signer = oci.config.from_file()

            self.object_storage_client = ObjectStorageClient(
                config={}
                if self.config.oracle_auth_type == "instance_principal"
                else signer,
                signer=signer
                if self.config.oracle_auth_type == "instance_principal"
                else None,
            )

            self.namespace = self.config.oracle_namespace
            self.bucket_name = self.config.oracle_bucket_name
            self.endpoint_url = self.config.get_oracle_endpoint()

            logger.info(f"Oracle Cloud Object Storage configured for Neural:")
            logger.info(f"  Namespace: {self.namespace}")
            logger.info(f"  Bucket: {self.bucket_name}")
            logger.info(f"  Region: {self.config.oracle_region}")
            logger.info(f"  Auth Type: {self.config.oracle_auth_type}")

        except Exception as e:
            logger.error(f"Error configurando Oracle Cloud Object Storage: {e}")
            raise

    def get_bucket_url(self, object_name: str) -> str:
        """Construye la URL completa del objeto en el bucket"""
        if self.storage_type == "oracle_cloud":
            return f"https://{self.namespace}.objectstorage.{self.config.oracle_region}.oci.customer-oci.com/n/{self.namespace}/b/{self.bucket_name}/o/{object_name}"
        else:
            return f"/evidencias_neural/{object_name}"

    async def save_image(
        self, image_data: bytes, imagename: str, plate_prefix: str = "unknown"
    ) -> Tuple[str, str]:
        """
        Guarda una imagen de Neural

        Args:
            image_data: Datos binarios de la imagen
            imagename: Nombre de la imagen
            plate_prefix: Prefijo de la placa para el nombre del archivo

        Returns:
            tuple: (ruta_relativa, url_completa_bucket)
        """
        if self.storage_type == "local":
            return await self._save_image_local(image_data, imagename, plate_prefix)
        elif self.storage_type == "oracle_cloud":
            return await self._save_image_oracle_cloud(
                image_data, imagename, plate_prefix
            )
        else:
            raise ValueError(
                f"Tipo de almacenamiento no soportado: {self.storage_type}"
            )

    async def _save_image_local(
        self, image_data: bytes, imagename: str, plate_prefix: str
    ) -> Tuple[str, str]:
        """Guarda imagen en almacenamiento local"""
        try:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            date_folder = self.base_folder / date
            date_folder.mkdir(exist_ok=True, parents=True)

            timestamp = datetime.now(timezone.utc).strftime("%H%M%S_%f")
            filename = f"{plate_prefix}_neural_{timestamp}_{imagename}"
            full_path = date_folder / filename

            async with aiofiles.open(full_path, "wb") as f:
                await f.write(image_data)

            relative_path = f"{date}/{filename}"
            full_bucket_url = self.get_bucket_url(
                f"evidencias_neural/{date}/{filename}"
            )
            logger.info(f"Neural image saved locally: {relative_path}")
            return relative_path, full_bucket_url

        except Exception as e:
            logger.error(f"Error saving Neural image locally: {e}")
            raise

    async def _save_image_oracle_cloud(
        self, image_data: bytes, imagename: str, plate_prefix: str
    ) -> Tuple[str, str]:
        """Guarda imagen en Oracle Cloud Object Storage"""
        try:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            timestamp = datetime.now(timezone.utc).strftime("%H%M%S_%f")
            filename = f"{plate_prefix}_neural_{timestamp}_{imagename}"
            # Path dentro del bucket: evidencias_neural/fecha/archivo.jpg
            object_name = f"evidencias_neural/{date}/{filename}"

            opc_meta = {
                "upload-time": datetime.now(timezone.utc).isoformat(),
                "worker-id": self.config.worker_id,
                "source": "neural-camera",
                "camera-type": "neural",
            }

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.object_storage_client.put_object(
                    namespace_name=self.namespace,
                    bucket_name=self.bucket_name,
                    object_name=object_name,
                    put_object_body=image_data,
                    content_type="image/jpeg",
                    opc_meta=opc_meta,
                ),
            )

            relative_path = f"{date}/{filename}"
            full_bucket_url = self.get_bucket_url(
                f"evidencias_neural/{date}/{filename}"
            )
            logger.info(f"Neural image uploaded to Oracle Cloud: {object_name}")
            return relative_path, full_bucket_url

        except ServiceError as e:
            logger.error(f"Oracle Cloud ServiceError: {e.message}")
            raise Exception(f"Failed to upload Neural image: {e.code}")
        except Exception as e:
            logger.error(f"Unexpected error uploading Neural image: {e}")
            raise

    async def get_image_url(
        self, image_path: str, expires_in: int = 3600
    ) -> Optional[str]:
        """Genera una URL pre-firmada para acceder a la imagen"""
        if self.storage_type == "local":
            return f"/evidencias_neural/{image_path}"

        elif self.storage_type == "oracle_cloud":
            try:
                object_name = f"evidencias_neural/{image_path}"

                loop = asyncio.get_event_loop()

                request = oci.object_storage.models.CreatePreauthenticatedRequestDetails(
                    name=f"neural-temp-access-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                    object_name=object_name,
                    access_type="ObjectRead",
                    time_expires=datetime.now(timezone.utc)
                    + timedelta(seconds=expires_in),
                )

                response = await loop.run_in_executor(
                    None,
                    lambda: self.object_storage_client.create_preauthenticated_request(
                        namespace_name=self.namespace,
                        bucket_name=self.bucket_name,
                        create_preauthenticated_request_details=request,
                    ),
                )

                base_url = f"https://objectstorage.{self.config.oracle_region}.oraclecloud.com"
                access_uri = response.data.access_uri
                full_url = f"{base_url}{access_uri}"

                return full_url

            except Exception as e:
                logger.error(
                    f"Error generating presigned URL for Neural image: {e}"
                )
                return None

        return None

    async def delete_image(self, image_path: str) -> bool:
        """Elimina una imagen del almacenamiento"""
        if self.storage_type == "local":
            return await self._delete_image_local(image_path)
        elif self.storage_type == "oracle_cloud":
            return await self._delete_image_oracle_cloud(image_path)
        return False

    async def _delete_image_local(self, image_path: str) -> bool:
        """Elimina imagen del almacenamiento local"""
        try:
            full_path = self.base_folder / image_path
            if full_path.exists():
                full_path.unlink()
                logger.info(f"Neural image deleted locally: {image_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting local Neural image: {e}")
            return False

    async def _delete_image_oracle_cloud(self, image_path: str) -> bool:
        """Elimina imagen de Oracle Cloud Object Storage"""
        try:
            object_name = f"evidencias_neural/{image_path}"

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.object_storage_client.delete_object(
                    namespace_name=self.namespace,
                    bucket_name=self.bucket_name,
                    object_name=object_name,
                ),
            )

            logger.info(f"Neural image deleted from Oracle Cloud: {object_name}")
            return True

        except Exception as e:
            logger.error(f"Error deleting Neural image from Oracle Cloud: {e}")
            return False

    async def health_check(self) -> dict:
        """Verifica el estado del servicio de almacenamiento"""
        if self.storage_type == "local":
            return {
                "storage_type": "local",
                "status": "healthy",
                "base_folder": str(self.base_folder),
                "folder_exists": self.base_folder.exists(),
                "bucket": "local_storage",
            }

        elif self.storage_type == "oracle_cloud":
            try:
                loop = asyncio.get_event_loop()

                response = await loop.run_in_executor(
                    None,
                    lambda: self.object_storage_client.get_bucket(
                        namespace_name=self.namespace,
                        bucket_name=self.bucket_name,
                    ),
                )

                return {
                    "storage_type": "oracle_cloud",
                    "status": "healthy",
                    "bucket": self.bucket_name,
                    "namespace": self.namespace,
                    "region": self.config.oracle_region,
                    "auth_type": self.config.oracle_auth_type,
                    "bucket_info": {
                        "name": response.data.name,
                        "compartment_id": response.data.compartment_id,
                        "created": response.data.time_created.isoformat()
                        if response.data.time_created
                        else None,
                    },
                }

            except Exception as e:
                return {
                    "storage_type": "oracle_cloud",
                    "status": "unhealthy",
                    "error": str(e),
                    "bucket": self.bucket_name,
                    "namespace": self.namespace,
                }

        return {"storage_type": "unknown", "status": "unhealthy"}
