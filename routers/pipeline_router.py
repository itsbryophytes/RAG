from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from models.schemas import DocumentUploadResponse
from models.enums import DocumentType
from pipelines.document_pipeline import ingest_document
from stores.pgvector_store import PGVectorStore
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])
_store = PGVectorStore()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    user_id: str = Form(..., min_length=1, max_length=128),
    document_type: DocumentType = Form(default=DocumentType.LAB_RESULT),
    file: UploadFile = File(...),
):
    logger.info(
        f"Upload request: user={user_id}, "
        f"file={file.filename}, type={document_type}"
    )
    return await ingest_document(user_id=user_id, file=file, document_type=document_type)


@router.get("/documents")
async def list_documents(
    user_id: str = Query(..., min_length=1, max_length=128)
):
    docs = await _store.list_documents(user_id)
    return {"user_id": user_id, "documents": docs, "count": len(docs)}


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    user_id: str = Query(..., min_length=1, max_length=128),
):
    deleted = await _store.delete_document(user_id=user_id, document_id=document_id)
    if deleted == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Document {document_id} not found for user {user_id}."
        )
    return {
        "message": f"Deleted {deleted} chunks for document {document_id}.",
        "document_id": document_id,
    }


@router.get("/health")
async def health():
    return {"status": "ok", "service": "pipeline"}