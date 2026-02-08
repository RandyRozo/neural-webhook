import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import asyncpg
import asyncpg.exceptions
from asyncpg import Pool

logger = logging.getLogger(__name__)


class NeuralDatabaseService:
    """
    Database service para eventos Neural con conexiones separadas de lectura/escritura
    """

    def __init__(self, config):
        self.config = config
        self.write_pool: Optional[Pool] = None
        self.read_pool: Optional[Pool] = None
        self._tables_ensured = False
        self._total_events = 0
        self.on_auth_error = None  # Callable[[Exception], Awaitable[bool]]

    @staticmethod
    def is_auth_error(exc: Exception) -> bool:
        """Detecta si una excepcion es un error de autenticacion de BD."""
        if isinstance(exc, asyncpg.exceptions.InvalidPasswordError):
            return True
        if isinstance(exc, asyncpg.exceptions.InvalidAuthorizationSpecificationError):
            return True
        error_msg = str(exc).lower()
        if "password authentication failed" in error_msg:
            return True
        if "authentication failed" in error_msg:
            return True
        return False

    async def initialize(self):
        """Initialize database connection pools"""
        try:
            # Create write connection pool
            self.write_pool = await asyncpg.create_pool(
                host=self.config.db_write_host,
                port=self.config.db_write_port,
                user=self.config.db_user,
                password=self.config.db_password,
                database=self.config.db_name,
                min_size=self.config.db_min_connections,
                max_size=self.config.db_max_connections,
                command_timeout=self.config.db_query_timeout,
                server_settings={
                    "application_name": f"neural_webhook_write_{self.config.worker_id}"
                },
            )
            logger.info(f"Created write connection pool to {self.config.db_write_host}")

            # Create read connection pool
            self.read_pool = await asyncpg.create_pool(
                host=self.config.db_read_host,
                port=self.config.db_read_port,
                user=self.config.db_user,
                password=self.config.db_password,
                database=self.config.db_name,
                min_size=self.config.db_min_connections,
                max_size=self.config.db_max_connections,
                command_timeout=self.config.db_query_timeout,
                server_settings={
                    "application_name": f"neural_webhook_read_{self.config.worker_id}"
                },
            )
            logger.info(f"Created read connection pool to {self.config.db_read_host}")

            # Ensure tables exist
            await self.ensure_tables_exist()

        except Exception as e:
            logger.error(f"Failed to initialize database service: {e}")
            raise

    async def ensure_tables_exist(self):
        """Create tables if they don't exist"""
        if self._tables_ensured:
            return

        try:
            async with self.write_pool.acquire() as conn:
                # Create detected_plates_wh_devices table (shared with other microservices)
                logger.info("Ensuring detected_plates_wh_devices table exists...")
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS detected_plates_wh_devices (
                        id SERIAL PRIMARY KEY,
                        plate VARCHAR(20),
                        image_url VARCHAR(500),
                        camera_brand VARCHAR(20) NOT NULL,
                        camera_id VARCHAR(100),
                        camera_location VARCHAR(200),
                        violation_type VARCHAR(100),
                        vehicle_type VARCHAR(50),
                        direction VARCHAR(20),
                        confidence DECIMAL(6,4),
                        capture_time TIMESTAMP WITH TIME ZONE,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT (NOW() AT TIME ZONE 'UTC'),
                        raw_data JSONB,
                        ocr_correction_report TEXT
                    )
                """)

                # Create indexes (IF NOT EXISTS for shared table)
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_wh_devices_plate ON detected_plates_wh_devices(plate)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_wh_devices_camera_id ON detected_plates_wh_devices(camera_id)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_wh_devices_created_at ON detected_plates_wh_devices(created_at)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_wh_devices_camera_brand ON detected_plates_wh_devices(camera_brand)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_wh_devices_capture_time ON detected_plates_wh_devices(capture_time)"
                )

                logger.info(
                    "detected_plates_wh_devices table and indexes ensured successfully"
                )

                # Tabla compartida de rechazos (protegida con IF NOT EXISTS)
                logger.info("Ensuring rejected_plates_wh_cameras table exists...")
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS rejected_plates_wh_cameras (
                        id SERIAL PRIMARY KEY,
                        camera_brand VARCHAR(20) NOT NULL,
                        camera_id VARCHAR(100),
                        raw_plate_text VARCHAR(50),
                        confidence DECIMAL(5,2),
                        rejection_reason TEXT NOT NULL,
                        rejection_type VARCHAR(50),
                        country VARCHAR(10),
                        vehicle_type VARCHAR(50),
                        raw_data TEXT,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT (NOW() AT TIME ZONE 'UTC')
                    )
                """)

                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_rejected_wh_brand_camera
                    ON rejected_plates_wh_cameras(camera_brand, camera_id, created_at DESC)
                """)
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_rejected_wh_type
                    ON rejected_plates_wh_cameras(rejection_type)
                """)
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_rejected_wh_confidence
                    ON rejected_plates_wh_cameras(confidence)
                """)

                logger.info("rejected_plates_wh_cameras table ensured successfully")

            self._tables_ensured = True

        except Exception as e:
            logger.error(f"Error ensuring tables exist: {e}")
            raise

    async def recreate_pools(self) -> None:
        """Cierra pools existentes y crea nuevos con las credenciales actuales del config.
        Usado tras refrescar la contrasena desde OCI Vault."""
        logger.info("Recreando pools de conexion con credenciales actualizadas...")

        if self.write_pool:
            await self.write_pool.close()
            logger.info("Pool de escritura anterior cerrado")
        if self.read_pool:
            await self.read_pool.close()
            logger.info("Pool de lectura anterior cerrado")

        self.write_pool = await asyncpg.create_pool(
            host=self.config.db_write_host,
            port=self.config.db_write_port,
            user=self.config.db_user,
            password=self.config.db_password,
            database=self.config.db_name,
            min_size=self.config.db_min_connections,
            max_size=self.config.db_max_connections,
            command_timeout=self.config.db_query_timeout,
            server_settings={
                "application_name": f"neural_webhook_write_{self.config.worker_id}"
            },
        )
        logger.info(f"Nuevo pool de escritura creado para {self.config.db_write_host}")

        self.read_pool = await asyncpg.create_pool(
            host=self.config.db_read_host,
            port=self.config.db_read_port,
            user=self.config.db_user,
            password=self.config.db_password,
            database=self.config.db_name,
            min_size=self.config.db_min_connections,
            max_size=self.config.db_max_connections,
            command_timeout=self.config.db_query_timeout,
            server_settings={
                "application_name": f"neural_webhook_read_{self.config.worker_id}"
            },
        )
        logger.info(f"Nuevo pool de lectura creado para {self.config.db_read_host}")

    async def _execute_save_event(self, event_data: dict) -> Optional[int]:
        """Implementacion interna de save_event."""
        async with self.write_pool.acquire() as conn:
            result = await conn.fetchval(
                """
                INSERT INTO detected_plates_wh_devices (
                    plate, image_url, camera_brand, camera_id, camera_location,
                    violation_type, vehicle_type, direction, confidence,
                    capture_time, created_at, raw_data, ocr_correction_report
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12::jsonb, $13)
                RETURNING id
            """,
                event_data.get("plate"),
                event_data.get("image_url"),
                event_data.get("camera_brand", "neural"),
                event_data.get("camera_id"),
                event_data.get("camera_location"),
                event_data.get("violation_type"),
                event_data.get("vehicle_type"),
                event_data.get("direction"),
                event_data.get("confidence"),
                event_data.get("capture_time"),
                datetime.now(timezone.utc),
                event_data.get("raw_data"),
                event_data.get("ocr_correction_report"),
            )

            if result:
                self._total_events += 1
                logger.info(
                    f"Neural event saved to detected_plates_wh_devices - ID: {result}, plate: {event_data.get('plate')}"
                )

            return result

    async def save_event(self, event_data: dict) -> Optional[int]:
        """Save a Neural event to the unified detected_plates_wh_devices table"""
        try:
            return await self._execute_save_event(event_data)
        except Exception as e:
            if self.is_auth_error(e) and self.on_auth_error:
                logger.warning(
                    "Error de autenticacion en save_event, refrescando credenciales..."
                )
                refreshed = await self.on_auth_error(e)
                if refreshed:
                    try:
                        return await self._execute_save_event(event_data)
                    except Exception as retry_e:
                        logger.error(
                            f"Error guardando evento tras refresh de credenciales: {retry_e}"
                        )
                        return None
            logger.error(f"Error saving Neural event to database: {e}")
            return None

    async def _execute_save_rejected(self, rejection_data: dict) -> Optional[int]:
        """Implementacion interna de save_rejected_plate_wh."""
        async with self.write_pool.acquire() as conn:
            result = await conn.fetchval(
                """
                INSERT INTO rejected_plates_wh_cameras (
                    camera_brand, camera_id, raw_plate_text, confidence,
                    rejection_reason, rejection_type, country, vehicle_type, raw_data
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING id
            """,
                rejection_data.get("camera_brand", "neural"),
                rejection_data.get("camera_id"),
                rejection_data.get("raw_plate_text"),
                rejection_data.get("confidence"),
                rejection_data.get("rejection_reason"),
                rejection_data.get("rejection_type"),
                rejection_data.get("country"),
                rejection_data.get("vehicle_type"),
                rejection_data.get("raw_data"),
            )

            if result:
                logger.info(
                    f"Rejected plate saved - Brand: {rejection_data.get('camera_brand')}, ID: {result}"
                )
            return result

    async def save_rejected_plate_wh(self, rejection_data: dict) -> Optional[int]:
        """Guarda placa rechazada en tabla compartida de webhooks"""
        try:
            return await self._execute_save_rejected(rejection_data)
        except Exception as e:
            if self.is_auth_error(e) and self.on_auth_error:
                logger.warning(
                    "Error de autenticacion en save_rejected_plate_wh, refrescando credenciales..."
                )
                refreshed = await self.on_auth_error(e)
                if refreshed:
                    try:
                        return await self._execute_save_rejected(rejection_data)
                    except Exception as retry_e:
                        logger.error(
                            f"Error guardando rechazo tras refresh de credenciales: {retry_e}"
                        )
                        return None
            logger.error(f"Error saving rejected plate: {e}")
            return None

    async def get_recent_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent Neural events from database"""
        try:
            async with self.read_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, plate, image_url, camera_brand, camera_id,
                           camera_location, violation_type, vehicle_type,
                           direction, confidence, capture_time, created_at, raw_data
                    FROM detected_plates_wh_devices
                    WHERE camera_brand = 'neural'
                    ORDER BY created_at DESC
                    LIMIT $1
                """,
                    limit,
                )

                events = []
                for row in rows:
                    events.append(
                        {
                            "id": row["id"],
                            "plate": row["plate"],
                            "confidence": float(row["confidence"])
                            if row["confidence"]
                            else None,
                            "capture_time": row["capture_time"],
                            "camera_id": row["camera_id"],
                            "camera_brand": row["camera_brand"],
                            "camera_location": row["camera_location"],
                            "image_url": row["image_url"],
                            "vehicle_type": row["vehicle_type"],
                            "direction": row["direction"],
                            "violation_type": row["violation_type"],
                            "created_at": row["created_at"],
                            "raw_data": row["raw_data"],
                        }
                    )

                return events

        except Exception as e:
            logger.error(f"Error getting recent Neural events: {e}")
            return []

    async def get_event_stats(self) -> Dict[str, Any]:
        """Get Neural event statistics"""
        try:
            async with self.read_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT
                        COUNT(*) as total_events,
                        COUNT(CASE WHEN DATE(created_at) = CURRENT_DATE THEN 1 END) as events_today,
                        COUNT(DISTINCT plate) FILTER (WHERE plate IS NOT NULL AND plate != '') as unique_plates,
                        COUNT(CASE WHEN created_at >= NOW() - INTERVAL '1 hour' THEN 1 END) as events_last_hour
                    FROM detected_plates_wh_devices
                    WHERE camera_brand = 'neural'
                """)

                return {
                    "total_events": row["total_events"] if row else 0,
                    "events_today": row["events_today"] if row else 0,
                    "unique_plates": row["unique_plates"] if row else 0,
                    "events_last_hour": row["events_last_hour"] if row else 0,
                }

        except Exception as e:
            logger.error(f"Error getting Neural event statistics: {e}")
            return {
                "total_events": 0,
                "events_today": 0,
                "unique_plates": 0,
                "events_last_hour": 0,
            }

    async def get_events_by_plate(
        self, plate: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get Neural events by plate number"""
        try:
            async with self.read_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, plate, image_url, camera_brand, camera_id,
                           camera_location, violation_type, vehicle_type,
                           direction, confidence, capture_time, created_at, raw_data
                    FROM detected_plates_wh_devices
                    WHERE plate = $1 AND camera_brand = 'neural'
                    ORDER BY created_at DESC
                    LIMIT $2
                """,
                    plate,
                    limit,
                )

                events = []
                for row in rows:
                    events.append(
                        {
                            "id": row["id"],
                            "plate": row["plate"],
                            "confidence": float(row["confidence"])
                            if row["confidence"]
                            else None,
                            "capture_time": row["capture_time"],
                            "camera_id": row["camera_id"],
                            "camera_brand": row["camera_brand"],
                            "camera_location": row["camera_location"],
                            "image_url": row["image_url"],
                            "vehicle_type": row["vehicle_type"],
                            "direction": row["direction"],
                            "violation_type": row["violation_type"],
                            "created_at": row["created_at"],
                            "raw_data": row["raw_data"],
                        }
                    )

                return events

        except Exception as e:
            logger.error(f"Error getting events by plate {plate}: {e}")
            return []

    @property
    def total_events_processed(self) -> int:
        """Get total events processed in this session"""
        return self._total_events

    async def close(self):
        """Close database connection pools"""
        try:
            if self.write_pool:
                await self.write_pool.close()
                logger.info("Write connection pool closed")

            if self.read_pool:
                await self.read_pool.close()
                logger.info("Read connection pool closed")

        except Exception as e:
            logger.error(f"Error closing database connections: {e}")
