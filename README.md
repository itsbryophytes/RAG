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