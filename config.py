from functools import lru_cache
from urllib.parse import quote

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GEMINI_API_KEY: str
    GEMINI_CHAT_MODEL: str = "gemini-2.5-flash"
    GEMINI_EMBED_MODEL: str = "models/gemini-embedding-2"
    GEMINI_EMBED_DIM: int = 768

    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str

    RETRIEVAL_TOP_K: int = 6
    RETRIEVAL_THRESHOLD: float = 0.65

    OCR_LANGUAGES: list[str] = ["en"]
    OCR_GPU: bool = False

    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    LOG_LEVEL: str = "INFO"
    ENVIRONMENT: str = "production"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def POSTGRES_DSN(self) -> str:
        user = quote(self.POSTGRES_USER, safe="")
        password = quote(self.POSTGRES_PASSWORD, safe="")
        database = quote(self.POSTGRES_DB, safe="")
        return (
            f"postgresql://{user}:{password}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{database}"
        )

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
