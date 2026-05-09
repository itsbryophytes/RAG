from __future__ import annotations

from typing import Any

from processing.chunker import chunk_text
from services.embedding_service import EmbeddingService
from stores.pgvector_store import PGVectorStore
from utils.logger import get_logger

logger = get_logger(__name__)
_embedding_svc = EmbeddingService()
_vector_store = PGVectorStore()


class RAGService:
    async def index_text(
        self,
        user_id: str,
        document_id: str,
        text: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> int:
        if not text.strip():
            logger.warning(f"index_text called with empty text for document={document_id}")
            return 0

        chunks = chunk_text(
            text=text,
            document_id=document_id,
            extra_metadata=extra_metadata or {},
        )
        if not chunks:
            logger.warning(f"Chunker produced 0 chunks for document={document_id}")
            return 0

        logger.info(f"RAG: {len(chunks)} chunks for document={document_id}")

        texts = [c.content for c in chunks]
        embeddings = await _embedding_svc.embed_documents_batch(texts)

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
        return await _vector_store.delete_document(user_id, document_id)