from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from routers import chat_router, pipeline_router
from stores.pgvector_store import PgVectorStore
from utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up Health Companion RAG API...")

    app.state.db_pool = await asyncpg.create_pool(
        dsn=settings.DATABASE_URL,
        min_size=5,
        max_size=20,
        command_timeout=60,
    )
    app.state.vector_store = PgVectorStore(pool=app.state.db_pool)
    logger.info("Database pool established.")

    yield

    logger.info("Shutting down — closing DB pool.")
    await app.state.db_pool.close()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router.router, prefix="/api/v1", tags=["Chat"])
app.include_router(pipeline_router.router, prefix="/api/v1", tags=["Pipeline"])


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok", "version": settings.APP_VERSION}