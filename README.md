# SETUP & MENJALANKAN

Menjalankan `pgvector`:
```bash
docker pull pgvector/pgvector:pg15
```

```bash
docker run -d --name pgvector -e POSTGRES_USER={USER} -e POSTGRES_PASSWORD={PASS} -e POSTGRES_DB=health_companion -p 5432:5432 pgvector/pgvector:pg15
```
Install dependencies:
```bash
pip install -r requirements.txt
```
Menjalankan `index.html`:
```bash
uvicorn main:app --reload
```

File `.env`:
```bash
GEMINI_API_KEY={GEMINI_API}

GEMINI_CHAT_MODEL=gemini-2.5-flash
GEMINI_EMBED_MODEL=models/gemini-embedding-2
GEMINI_EMBED_DIM=3072

POSTGRES_DSN=postgresql://{USER_POSTGRESQL}:{PASSWORD_POSTGRESQL}@localhost:5432/health_companion

RETRIEVAL_TOP_K=6
RETRIEVAL_THRESHOLD=0.65

OCR_LANGUAGES=["en"]
OCR_GPU=false

CHUNK_SIZE=512
CHUNK_OVERLAP=64

LOG_LEVEL=INFO
ENVIRONMENT=development
```