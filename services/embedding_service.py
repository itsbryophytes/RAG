import asyncio
from functools import lru_cache

from google import genai
from google.genai import types

from config import get_settings
from utils.logger import get_logger
from utils.retry import async_retry

logger = get_logger(__name__)
settings = get_settings()

_client = None

@lru_cache(maxsize=1)
def _configure_genai():
    global _client
    _client = genai.Client(api_key=settings.GEMINI_API_KEY)


class EmbeddingService:
    def __init__(self):
        _configure_genai()

        self.client = _client
        self.model = settings.GEMINI_EMBED_MODEL
        self.dim = settings.GEMINI_EMBED_DIM

    @async_retry(max_attempts=3, base_delay=1.0, exceptions=(Exception,))
    async def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Empty text")

        loop = asyncio.get_event_loop()

        result = await loop.run_in_executor(
            None,
            lambda: self.client.models.embed_content(
                model=self.model,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                ),
            ),
        )

        embedding = result.embeddings[0].values
        self._validate_dim(embedding)
        return embedding

    @async_retry(max_attempts=3, base_delay=1.0, exceptions=(Exception,))
    async def embed_document(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Empty chunk")

        loop = asyncio.get_event_loop()

        result = await loop.run_in_executor(
            None,
            lambda: self.client.models.embed_content(
                model=self.model,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                ),
            ),
        )

        embedding = result.embeddings[0].values
        self._validate_dim(embedding)
        return embedding

    async def embed_documents_batch(self, texts: list[str], batch_size: int = 10):
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            tasks = [self.embed_document(t) for t in batch]
            results = await asyncio.gather(*tasks)

            all_embeddings.extend(results)

            logger.info(
                f"Embedded batch {i // batch_size + 1} "
                f"({len(all_embeddings)}/{len(texts)})"
            )

        return all_embeddings

    def _validate_dim(self, embedding: list[float]):
        if len(embedding) != self.dim:
            raise ValueError(
                f"Embedding dim mismatch: expected {self.dim}, got {len(embedding)}"
            )