from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import get_settings
from stores.pgvector_store import PGVectorStore
from stores.staging_store import StagingStore
from routers import chat_router, pipeline_router
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Health RAG service...")

    store = PGVectorStore()
    await store.init_db()
    staging = StagingStore()
    await staging.init_db()
    logger.info("Database ready.")

    yield

    logger.info("Shutting down Health RAG service.")


app = FastAPI(
    title="Health Companion RAG API",
    version="1.0.0",
    description=(
        "Production RAG service for the Health Companion app. "
        "Processes medical documents, extracts lab parameters, "
        "and answers health questions using retrieved context."
    ),
    lifespan=lifespan,
    docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.ENVIRONMENT == "development" else [""],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "PUT", "PATCH"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again."},
    )


app.include_router(chat_router.router)
app.include_router(pipeline_router.router)

@app.get("/")
async def root():
    return {
        "service": "Health Companion RAG API",
        "version": "1.0.0",
        "docs": "/docs",
    }

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}