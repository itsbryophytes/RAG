from __future__ import annotations

import json
from typing import Any

import asyncpg

from config import get_settings
from models.schemas import RetrievedChunk
from utils.logger import get_logger
from utils.retry import async_retry

logger = get_logger(__name__)
settings = get_settings()

_POOL: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    global _POOL
    if _POOL is None:
        _POOL = await asyncpg.create_pool(
            dsn=settings.POSTGRES_DSN,
            min_size=2,
            max_size=10,
            command_timeout=30,
            init=_init_connection,
        )
    return _POOL


async def _init_connection(conn: asyncpg.Connection) -> None:
    try:
        await conn.execute('CREATE EXTENSION IF NOT EXISTS vector;')
    except Exception as e:
        logger.warning(f"vector extension issue ignored: {e}")


def _encode_vector(vec: list[float]) -> str:
    return "[" + ",".join(str(v) for v in vec) + "]"


class PGVectorStore:
    async def init_db(self) -> None:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                CREATE EXTENSION IF NOT EXISTS vector;
                CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

                CREATE TABLE IF NOT EXISTS document_chunks (
                    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id     TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    chunk_id    TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    embedding   VECTOR({settings.GEMINI_EMBED_DIM}) NOT NULL,
                    metadata    JSONB DEFAULT '{{}}',
                    created_at  TIMESTAMPTZ DEFAULT NOW(),
                    CONSTRAINT uq_chunk_id UNIQUE (chunk_id)
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_user_id
                    ON document_chunks (user_id);

                CREATE INDEX IF NOT EXISTS idx_chunks_embedding
                    ON document_chunks
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                """
            )
            logger.info("pgvector schema initialised.")

    @async_retry(max_attempts=3, base_delay=0.5, exceptions=(asyncpg.PostgresError,))
    async def upsert_chunk(
        self,
        user_id: str,
        document_id: str,
        chunk_id: str,
        content: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO document_chunks
                    (user_id, document_id, chunk_id, content, embedding, metadata)
                VALUES ($1, $2, $3, $4, $5::VECTOR, $6::JSONB)
                ON CONFLICT (chunk_id)
                DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata;
                """,
                user_id,
                document_id,
                chunk_id,
                content,
                _encode_vector(embedding),
                json.dumps(metadata or {}),
            )

    async def upsert_chunks_bulk(
        self,
        user_id: str,
        document_id: str,
        chunks: list[dict],
    ) -> int:
        if not chunks:
            return 0

        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(
                    """
                    INSERT INTO document_chunks
                        (user_id, document_id, chunk_id, content, embedding, metadata)
                    VALUES ($1, $2, $3, $4, $5::VECTOR, $6::JSONB)
                    ON CONFLICT (chunk_id)
                    DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata;
                    """,
                    [
                        (
                            user_id,
                            document_id,
                            c["chunk_id"],
                            c["content"],
                            _encode_vector(c["embedding"]),
                            json.dumps(c.get("metadata", {})),
                        )
                        for c in chunks
                    ],
                )

        logger.info(
            f"Bulk upserted {len(chunks)} chunks for "
            f"document={document_id} user={user_id}"
        )
        return len(chunks)

    @async_retry(max_attempts=3, base_delay=0.5, exceptions=(asyncpg.PostgresError,))
    async def similarity_search(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int | None = None,
        threshold: float | None = None,
        metadata_filter: dict | None = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or settings.RETRIEVAL_TOP_K
        threshold = (
            threshold
            if threshold is not None
            else settings.RETRIEVAL_THRESHOLD
        )

        pool = await get_pool()
        async with pool.acquire() as conn:
            if metadata_filter:
                rows = await conn.fetch(
                    """
                    SELECT
                        chunk_id,
                        content,
                        document_id,
                        metadata,
                        1 - (embedding <=> $1::VECTOR) AS score
                    FROM document_chunks
                    WHERE user_id = $2
                      AND 1 - (embedding <=> $1::VECTOR) >= $3
                      AND metadata @> $5::jsonb
                    ORDER BY embedding <=> $1::VECTOR
                    LIMIT $4;
                    """,
                    _encode_vector(query_embedding),
                    user_id,
                    threshold,
                    top_k,
                    json.dumps(metadata_filter),
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT
                        chunk_id,
                        content,
                        document_id,
                        metadata,
                        1 - (embedding <=> $1::VECTOR) AS score
                    FROM document_chunks
                    WHERE user_id = $2
                      AND 1 - (embedding <=> $1::VECTOR) >= $3
                    ORDER BY embedding <=> $1::VECTOR
                    LIMIT $4;
                    """,
                    _encode_vector(query_embedding),
                    user_id,
                    threshold,
                    top_k,
                )

        results = [
            RetrievedChunk(
                chunk_id=row["chunk_id"],
                content=row["content"],
                score=float(row["score"]),
                source_document_id=row["document_id"],
                metadata=json.loads(row["metadata"]) if isinstance(row["metadata"], str) else (row["metadata"] or {}),
            )
            for row in rows
        ]

        logger.info(
            f"Similarity search: user={user_id}, "
            f"threshold={threshold}, top_k={top_k}, "
            f"returned={len(results)} chunks"
        )
        return results

    async def delete_document(self, user_id: str, document_id: str) -> int:
        pool = await get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM document_chunks
                WHERE user_id = $1 AND document_id = $2;
                """,
                user_id,
                document_id,
            )

        count = int(result.split()[-1])
        logger.info(
            f"Deleted {count} chunks for document={document_id} user={user_id}"
        )
        return count

    async def list_documents(self, user_id: str) -> list[dict]:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT
                    document_id,
                    MIN(created_at) AS uploaded_at,
                    COUNT(*) AS chunk_count,
                    (array_agg(metadata))[1] AS sample_meta
                FROM document_chunks
                WHERE user_id = $1
                GROUP BY document_id
                ORDER BY uploaded_at DESC;
                """,
                user_id,
            )

        return [dict(row) for row in rows]
