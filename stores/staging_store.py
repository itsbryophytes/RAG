from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

import asyncpg

from stores.pgvector_store import get_pool
from models.enums import DocumentType, StagingStatus
from utils.logger import get_logger
from utils.retry import async_retry

logger = get_logger(__name__)

STAGING_TTL_HOURS = 24


class StagingStore:
    async def init_db(self) -> None:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS lab_result_staging (
                    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id     TEXT NOT NULL UNIQUE,
                    user_id         TEXT NOT NULL,
                    filename        TEXT NOT NULL DEFAULT '',
                    document_type   TEXT NOT NULL DEFAULT 'lab_result',
                    structured_data JSONB,
                    ocr_confidence  FLOAT DEFAULT 0,
                    status          TEXT NOT NULL DEFAULT 'pending',
                    created_at      TIMESTAMPTZ DEFAULT NOW(),
                    expires_at      TIMESTAMPTZ NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_staging_user_id
                    ON lab_result_staging (user_id);

                CREATE INDEX IF NOT EXISTS idx_staging_status
                    ON lab_result_staging (status);

                CREATE TABLE IF NOT EXISTS lab_results (
                    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id     TEXT NOT NULL UNIQUE,
                    user_id         TEXT NOT NULL,
                    filename        TEXT NOT NULL DEFAULT '',
                    document_type   TEXT NOT NULL DEFAULT 'lab_result',
                    structured_data JSONB,
                    ocr_confidence  FLOAT DEFAULT 0,
                    confirmed_at    TIMESTAMPTZ DEFAULT NOW()
                );

                CREATE INDEX IF NOT EXISTS idx_lab_results_user_id
                    ON lab_results (user_id);
            """)
            logger.info("Staging schema initialised.")

    @async_retry(max_attempts=3, base_delay=0.5, exceptions=(asyncpg.PostgresError,))
    async def save_staging(
        self,
        document_id: str,
        user_id: str,
        filename: str,
        document_type: DocumentType,
        structured_data: Optional[dict[str, Any]],
        ocr_confidence: float,
    ) -> None:
        expires_at = datetime.now(timezone.utc) + timedelta(hours=STAGING_TTL_HOURS)
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO lab_result_staging
                    (document_id, user_id, filename, document_type,
                     structured_data, ocr_confidence, status, expires_at)
                VALUES ($1, $2, $3, $4, $5::JSONB, $6, 'pending', $7)
                ON CONFLICT (document_id) DO UPDATE SET
                    structured_data = EXCLUDED.structured_data,
                    ocr_confidence  = EXCLUDED.ocr_confidence,
                    status          = 'pending',
                    expires_at      = EXCLUDED.expires_at;
                """,
                document_id,
                user_id,
                filename,
                document_type.value,
                json.dumps(structured_data) if structured_data else None,
                ocr_confidence,
                expires_at,
            )
        logger.info(f"Staging saved: document={document_id} user={user_id}")

    async def get_staging(
        self, document_id: str, user_id: str
    ) -> Optional[dict[str, Any]]:
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT document_id, user_id, filename, document_type,
                       structured_data, ocr_confidence, status,
                       created_at, expires_at
                FROM lab_result_staging
                WHERE document_id = $1
                  AND user_id     = $2
                  AND expires_at  > NOW()
                  AND status      = 'pending';
                """,
                document_id,
                user_id,
            )
        if not row:
            return None
        return {
            "document_id":    row["document_id"],
            "user_id":        row["user_id"],
            "filename":       row["filename"],
            "document_type":  row["document_type"],
            "structured_data": json.loads(row["structured_data"]) if row["structured_data"] else None,
            "ocr_confidence": row["ocr_confidence"],
            "status":         row["status"],
            "created_at":     row["created_at"].isoformat(),
            "expires_at":     row["expires_at"].isoformat(),
        }

    async def list_pending(self, user_id: str) -> list[dict[str, Any]]:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT document_id, filename, document_type,
                       ocr_confidence, created_at, expires_at
                FROM lab_result_staging
                WHERE user_id   = $1
                  AND status    = 'pending'
                  AND expires_at > NOW()
                ORDER BY created_at DESC;
                """,
                user_id,
            )
        return [dict(r) for r in rows]


    @async_retry(max_attempts=3, base_delay=0.5, exceptions=(asyncpg.PostgresError,))
    async def confirm(self, document_id: str, user_id: str, new_data: Optional[dict[str, Any]] = None) -> bool:
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT document_id, user_id, filename, document_type,
                       structured_data, ocr_confidence
                FROM lab_result_staging
                WHERE document_id = $1
                  AND user_id     = $2
                  AND status      = 'pending'
                  AND expires_at  > NOW();
                """,
                document_id, user_id,
            )
            if not row:
                logger.warning(
                    f"Confirm: staging record not found or expired "
                    f"document={document_id} user={user_id}"
                )
                return False

            final_structured = new_data if new_data is not None else json.loads(row["structured_data"])

            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO lab_results
                        (document_id, user_id, filename, document_type,
                         structured_data, ocr_confidence)
                    VALUES ($1, $2, $3, $4, $5::JSONB, $6)
                    ON CONFLICT (document_id) DO UPDATE SET
                        structured_data = EXCLUDED.structured_data,
                        confirmed_at    = NOW();
                    """,
                    row["document_id"], row["user_id"],
                    row["filename"], row["document_type"],
                    json.dumps(final_structured), row["ocr_confidence"],
                )
                await conn.execute(
                    """
                    UPDATE lab_result_staging
                    SET status = 'confirmed'
                    WHERE document_id = $1 AND user_id = $2;
                    """,
                    document_id, user_id,
                )

        logger.info(f"Confirmed: document={document_id} user={user_id}")
        return True

    @async_retry(max_attempts=3, base_delay=0.5, exceptions=(asyncpg.PostgresError,))
    async def discard(self, document_id: str, user_id: str) -> bool:
        pool = await get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE lab_result_staging
                SET status = 'discarded'
                WHERE document_id = $1
                  AND user_id     = $2
                  AND status      = 'pending';
                """,
                document_id, user_id,
            )
        found = int(result.split()[-1]) > 0
        if found:
            logger.info(f"Discarded: document={document_id} user={user_id}")
        else:
            logger.warning(
                f"Discard: no pending record for document={document_id} user={user_id}"
            )
        return found

    async def cleanup_expired(self) -> int:
        pool = await get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM lab_result_staging WHERE expires_at <= NOW();"
            )
        count = int(result.split()[-1])
        if count:
            logger.info(f"Cleaned up {count} expired staging records.")
        return count

    async def get_lab_results(self, user_id: str) -> list[dict[str, Any]]:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT document_id, filename, document_type,
                       structured_data, ocr_confidence, confirmed_at
                FROM lab_results
                WHERE user_id = $1
                ORDER BY confirmed_at DESC;
                """,
                user_id,
            )
        return [
            {
                **dict(r),
                "structured_data": self._normalize_structured_data(r["structured_data"]),
            }
            for r in rows
        ]

    @async_retry(max_attempts=3, base_delay=0.5, exceptions=(asyncpg.PostgresError,))
    async def update_result(self, document_id: str, user_id: str, new_data: dict[str, Any]) -> bool:
        pool = await get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE lab_results
                SET structured_data = $3::JSONB
                WHERE document_id = $1 AND user_id = $2;
                """,
                document_id, user_id, json.dumps(new_data)
            )
        found = int(result.split()[-1]) > 0
        if found:
            logger.info(f"Updated result: document={document_id} user={user_id}")
        return found

    @async_retry(max_attempts=3, base_delay=0.5, exceptions=(asyncpg.PostgresError,))
    async def delete_result(self, document_id: str, user_id: str) -> bool:
        pool = await get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM lab_results WHERE document_id = $1 AND user_id = $2;",
                document_id, user_id
            )
        found = int(result.split()[-1]) > 0
        if found:
            logger.info(f"Deleted result: document={document_id} user={user_id}")
        return found

    @async_retry(max_attempts=3, base_delay=0.5, exceptions=(asyncpg.PostgresError,))
    async def save_manual(
        self,
        document_id: str,
        user_id: str,
        filename: str,
        document_type: DocumentType,
        structured_data: dict[str, Any],
    ) -> None:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO lab_results
                    (document_id, user_id, filename, document_type,
                     structured_data, ocr_confidence)
                VALUES ($1, $2, $3, $4, $5::JSONB, $6)
                ON CONFLICT (document_id) DO UPDATE SET
                    structured_data = EXCLUDED.structured_data,
                    confirmed_at    = NOW();
                """,
                document_id,
                user_id,
                filename,
                document_type.value,
                json.dumps(structured_data),
                1.0,
            )
        logger.info(f"Manual record saved: document={document_id} user={user_id}")
