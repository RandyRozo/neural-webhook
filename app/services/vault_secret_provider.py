import asyncio
import base64
import logging
import time
from typing import Dict, Optional, Tuple

import oci
from oci.exceptions import ServiceError
from oci.secrets import SecretsClient

logger = logging.getLogger(__name__)

DEFAULT_CACHE_TTL_SECONDS = 24 * 60 * 60  # 24 horas


class VaultSecretProvider:
    """
    Obtiene secretos de OCI Vault usando Instance Principal.
    Cache en memoria con TTL configurable (default 24 horas).
    Fallback a variables de entorno cuando Vault no esta disponible.
    """

    def __init__(
        self,
        vault_id: str,
        auth_type: str = "instance_principal",
        cache_ttl: int = DEFAULT_CACHE_TTL_SECONDS,
    ):
        self.vault_id = vault_id
        self.auth_type = auth_type
        self._cache_ttl = cache_ttl
        self._secrets_client: Optional[SecretsClient] = None
        self._signer = None
        self._cache: Dict[str, Tuple[str, float]] = {}
        self._initialized = False

    def initialize(self) -> None:
        """
        Inicializa el SecretsClient de OCI. Llamada sincrona al inicio.
        """
        try:
            if self.auth_type == "instance_principal":
                logger.info(
                    "VaultSecretProvider: Inicializando Instance Principal signer..."
                )
                self._signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
                self._secrets_client = SecretsClient(config={}, signer=self._signer)
            else:
                logger.info(
                    "VaultSecretProvider: Inicializando signer desde config file..."
                )
                config = oci.config.from_file()
                self._secrets_client = SecretsClient(config)

            self._initialized = True
            logger.info("VaultSecretProvider: Inicializado exitosamente")
        except Exception as e:
            logger.error(f"VaultSecretProvider: Error al inicializar: {e}")
            raise

    async def get_secret(self, secret_name: str, force_refresh: bool = False) -> str:
        """
        Obtiene el valor de un secreto por nombre desde OCI Vault.
        Usa cache si esta disponible y no ha expirado.
        """
        if not self._initialized:
            raise RuntimeError("VaultSecretProvider no inicializado")

        if not force_refresh and secret_name in self._cache:
            value, fetched_at = self._cache[secret_name]
            age = time.time() - fetched_at
            if age < self._cache_ttl:
                logger.debug(
                    f"VaultSecretProvider: Cache hit para '{secret_name}' "
                    f"(edad: {age:.0f}s)"
                )
                return value
            else:
                logger.info(
                    f"VaultSecretProvider: Cache expirado para '{secret_name}' "
                    f"(edad: {age:.0f}s)"
                )

        return await self._fetch_and_cache(secret_name)

    async def _fetch_and_cache(self, secret_name: str) -> str:
        """
        Obtiene el secreto desde OCI Vault (llamada sincrona en executor)
        y lo almacena en cache.
        """
        logger.info(
            f"VaultSecretProvider: Obteniendo secreto '{secret_name}' desde vault..."
        )

        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: self._secrets_client.get_secret_bundle_by_name(
                    secret_name=secret_name, vault_id=self.vault_id
                ),
            )

            base64_content = response.data.secret_bundle_content.content
            secret_value = base64.b64decode(base64_content).decode("utf-8")

            self._cache[secret_name] = (secret_value, time.time())
            logger.info(
                f"VaultSecretProvider: Secreto '{secret_name}' obtenido y cacheado"
            )

            return secret_value

        except ServiceError as e:
            if e.status in (401, 403):
                logger.warning(
                    f"VaultSecretProvider: Error de auth ({e.status}) obteniendo "
                    f"'{secret_name}', refrescando signer..."
                )
                await self._refresh_signer()
                response = await loop.run_in_executor(
                    None,
                    lambda: self._secrets_client.get_secret_bundle_by_name(
                        secret_name=secret_name, vault_id=self.vault_id
                    ),
                )
                base64_content = response.data.secret_bundle_content.content
                secret_value = base64.b64decode(base64_content).decode("utf-8")
                self._cache[secret_name] = (secret_value, time.time())
                logger.info(
                    f"VaultSecretProvider: Secreto '{secret_name}' obtenido tras "
                    f"refresh del signer"
                )
                return secret_value
            else:
                logger.error(
                    f"VaultSecretProvider: ServiceError obteniendo '{secret_name}': "
                    f"{e.status} - {e.message}"
                )
                raise

    async def _refresh_signer(self) -> None:
        """
        Refresca el Instance Principal signer.
        """
        if self.auth_type != "instance_principal":
            return

        logger.info("VaultSecretProvider: Refrescando Instance Principal signer...")
        loop = asyncio.get_event_loop()
        try:
            self._signer = await loop.run_in_executor(
                None, oci.auth.signers.InstancePrincipalsSecurityTokenSigner
            )
            self._secrets_client = SecretsClient(config={}, signer=self._signer)
            logger.info("VaultSecretProvider: Signer refrescado exitosamente")
        except Exception as e:
            logger.error(f"VaultSecretProvider: Error al refrescar signer: {e}")
            raise

    def invalidate(self, secret_name: str) -> None:
        """Elimina un secreto especifico del cache."""
        if secret_name in self._cache:
            del self._cache[secret_name]
            logger.info(f"VaultSecretProvider: Cache invalidado para '{secret_name}'")

    def invalidate_all(self) -> None:
        """Limpia todo el cache de secretos."""
        self._cache.clear()
        logger.info("VaultSecretProvider: Todo el cache de secretos invalidado")
