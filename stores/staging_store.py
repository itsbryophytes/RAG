"""
stores/staging_store.py — Staging layer for structured lab data.

WHY THIS EXISTS:
  After OCR we have two outputs:
    A) clean text  → goes to pgvector immediately (RAG is ready)
    B) structured  → stays here until user confirms

  This table is the "draft" state. Nothing in here is part of the
  user's permanent health record until they explicitly confirm.

TABLES:
  lab_result_staging  — temporary, expires 24h after upload
  lab_results         — permanent, written only on confirmation

Both tables live in the same PostgreSQL database as pgvector.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

import asyncpg

from stores.pgvector_store import get_pool   # reuse the same connection pool
from models.enums import DocumentType, StagingStatus
from utils.logger import get_logger
from utils.retry import async_retry

logger = get_logger(__name__)

STAGING_TTL_HOURS = 24   # staging records auto-expire after this


class StagingStore:
    """
    Manages the staging → confirm/discard lifecycle for structured lab data.
    Uses the same asyncpg pool as PGVectorStore.
    """

    async def init_db(self) -> None:
        """Create staging and final tables if they don't exist. Idempotent."""
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

    # ── Write staging ─────────────────────────────────────────────

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
        """
        Save structured data to staging (pending state).
        The record expires automatically after STAGING_TTL_HOURS.
        """
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

    # ── Read staging ──────────────────────────────────────────────

    async def get_staging(
        self, document_id: str, user_id: str
    ) -> Optional[dict[str, Any]]:
        """
        Retrieve a staging record.
        Returns None if not found, expired, or wrong user_id.
        """
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
        """List all pending staging records for a user."""
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

    # ── Confirm ───────────────────────────────────────────────────

    @async_retry(max_attempts=3, base_delay=0.5, exceptions=(asyncpg.PostgresError,))
    async def confirm(self, document_id: str, user_id: str, updated_data: dict | None = None) -> bool:
        """
        Move structured data from staging → lab_results (permanent).
        If updated_data is provided, it replaces what was in staging.
        """
        pool = await get_pool()
        async with pool.acquire() as conn:
            # Fetch the staging row (verify ownership)
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

            final_data = json.dumps(updated_data) if updated_data else row["structured_data"]

            async with conn.transaction():
                # Insert into permanent table
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
                    final_data, row["ocr_confidence"],
                )
                # Mark staging as confirmed (don't delete — useful for audit)
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

    # ── Discard ───────────────────────────────────────────────────

    @async_retry(max_attempts=3, base_delay=0.5, exceptions=(asyncpg.PostgresError,))
    async def discard(self, document_id: str, user_id: str) -> bool:
        """
        Mark staging record as discarded.
        The caller (router) is responsible for also removing RAG vectors.
        Returns True if a record was found.
        """
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

    # ── Cleanup ───────────────────────────────────────────────────

    async def cleanup_expired(self) -> int:
        """
        Delete staging records past their TTL.
        Call from a background task or cron — not on every request.
        """
        pool = await get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM lab_result_staging WHERE expires_at <= NOW();"
            )
        count = int(result.split()[-1])
        if count:
            logger.info(f"Cleaned up {count} expired staging records.")
        return count

    # ── Query confirmed results ────────────────────────────────────

    @staticmethod
    def _normalize_structured_data(raw: str | dict | None) -> dict:
        """
        Normalize structured_data to a single canonical shape:
            { report_date, lab_name, metrics: { snake_case_key: value_string } }

        Handles three historical storage formats:
        1. New format  – { report_date, lab_name, metrics: {...} }
        2. Manual v1   – { date, lab_name, metrics: {...} }
        3. Raw OCR     – { date, lab_name, parameters: { "Display Name": { value, unit } } }
        4. NULL / empty
        """
        if not raw:
            return {"report_date": "", "lab_name": "", "metrics": {}}

        sd: dict = json.loads(raw) if isinstance(raw, str) else raw

        # ── Extract scalar meta fields ──────────────────────────────
        report_date: str = (
            sd.get("report_date")
            or sd.get("date")
            or ""
        )
        lab_name: str = sd.get("lab_name") or sd.get("lab") or ""

        # ── Build canonical metrics dict ────────────────────────────
        # Prefer already-normalised metrics block (new format / manual)
        metrics: dict = {}

        if sd.get("metrics"):
            raw_metrics = sd["metrics"]
            for k, v in raw_metrics.items():
                if k == "additional_metrics":
                    continue
                if isinstance(v, dict) and "value" in v:
                    metrics[k] = str(v["value"]) if v["value"] is not None else ""
                elif v is not None:
                    metrics[k] = str(v)

            # Preserve additional_metrics sub-dict as-is
            if "additional_metrics" in raw_metrics:
                metrics["additional_metrics"] = raw_metrics["additional_metrics"]

        elif sd.get("parameters"):
            # Raw OCR format: keys are display names like "Glucose Fasting"
            _DISPLAY_TO_SNAKE: dict[str, str] = {
                "glucose fasting": "glucose_fasting",
                "fasting glucose": "glucose_fasting",
                "gula puasa": "glucose_fasting",
                "glucose postmeal": "glucose_postmeal",
                "post-meal glucose": "glucose_postmeal",
                "post meal glucose": "glucose_postmeal",
                "hba1c": "hba1c",
                "hemoglobin a1c": "hba1c",
                "total cholesterol": "chol_total",
                "cholesterol total": "chol_total",
                "ldl": "chol_ldl",
                "ldl cholesterol": "chol_ldl",
                "hdl": "chol_hdl",
                "hdl cholesterol": "chol_hdl",
                "triglyceride": "triglycerides",
                "triglycerides": "triglycerides",
                "hemoglobin": "hemoglobin",
                "hematocrit": "hematocrit",
                "white blood": "wbc",
                "leukosit": "wbc",
                "wbc": "wbc",
                "platelet": "platelets",
                "trombosit": "platelets",
                "uric acid": "uric_acid",
                "asam urat": "uric_acid",
                "creatinine": "creatinine",
                "kreatinin": "creatinine",
                "blood urea nitrogen": "bun",
                "bun": "bun",
            }

            for display_name, param in sd["parameters"].items():
                key_lower = display_name.lower()
                snake = next(
                    (v for k, v in _DISPLAY_TO_SNAKE.items() if k in key_lower),
                    None,
                )
                if snake is None:
                    snake = display_name  # keep unknown keys as-is
                value = param.get("value") if isinstance(param, dict) else param
                metrics[snake] = str(value) if value is not None else ""

        return {
            "report_date": report_date,
            "lab_name": lab_name,
            "metrics": metrics,
        }

    async def get_lab_results(self, user_id: str) -> list[dict[str, Any]]:
        """Retrieve all confirmed lab results for dashboard."""
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

    async def update_lab_result(self, document_id: str, user_id: str, updated_data: dict[str, Any]) -> bool:
        """Update an existing confirmed lab result."""
        pool = await get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE lab_results
                SET structured_data = $1::JSONB
                WHERE document_id = $2 AND user_id = $3;
                """,
                json.dumps(updated_data),
                document_id,
                user_id,
            )
        found = int(result.split()[-1]) > 0
        if found:
            logger.info(f"Updated: document={document_id} user={user_id}")
        return found