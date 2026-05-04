"""
services/rag_service.py — RAG indexing service.

Single responsibility: take clean text → chunk → embed → store in pgvector.

This was previously scattered across document_pipeline.py.
Extracting it here lets the ingestion pipeline call it independently
and makes the chat pipeline testable without re-running OCR.
"""

from __future__ import annotations

from typing import Any

from processing.chunker import chunk_text
from services.embedding_service import EmbeddingService
from stores.pgvector_store import PGVectorStore
from utils.logger import get_logger

logger = get_logger(__name__)

# Singletons — initialised once per process
_embedding_svc = EmbeddingService()
_vector_store = PGVectorStore()


class RAGService:
    """Chunk → embed → upsert pipeline wrapper."""

    async def index_text(
        self,
        user_id: str,
        document_id: str,
        text: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> int:
        """
        Index a block of clean text into the vector store.

        Steps:
          1. Chunk the text
          2. Embed all chunks in batch
          3. Bulk upsert to pgvector

        Returns number of chunks indexed.
        Raises on embedding or DB failure (caller should catch and handle).
        """
        if not text.strip():
            logger.warning(f"index_text called with empty text for document={document_id}")
            return 0

        # ── 1. Chunk ──────────────────────────────────────────────
        chunks = chunk_text(
            text=text,
            document_id=document_id,
            extra_metadata=extra_metadata or {},
        )
        if not chunks:
            logger.warning(f"Chunker produced 0 chunks for document={document_id}")
            return 0

        logger.info(f"RAG: {len(chunks)} chunks for document={document_id}")

        # ── 2. Embed ──────────────────────────────────────────────
        texts = [c.content for c in chunks]
        embeddings = await _embedding_svc.embed_documents_batch(texts)

        # ── 3. Upsert ─────────────────────────────────────────────
        chunk_records = [
            {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "embedding": embeddings[i],
                "metadata": chunk.metadata,
            }
            for i, chunk in enumerate(chunks)
        ]
        await _vector_store.upsert_chunks_bulk(user_id, document_id, chunk_records)

        logger.info(
            f"RAG indexed: user={user_id}, document={document_id}, "
            f"chunks={len(chunks)}"
        )
        return len(chunks)

    async def delete_document(self, user_id: str, document_id: str) -> int:
        """Remove all vector chunks for a document (used on discard)."""
        return await _vector_store.delete_document(user_id, document_id)