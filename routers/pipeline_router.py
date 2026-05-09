from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, Request
import uuid

from models.schemas import IngestionResponse, ConfirmResponse, DiscardResponse, UpdateResultRequest
from models.enums import DocumentType
from pipelines.ingestion_pipeline import ingest
from services.rag_service import RAGService
from stores.staging_store import StagingStore
from stores.pgvector_store import PGVectorStore
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

_staging = StagingStore()
_rag_svc = RAGService()
_vector_store = PGVectorStore()


@router.post("/upload", response_model=IngestionResponse)
async def upload_document(
    user_id: str = Form(..., min_length=1, max_length=128),
    document_type: DocumentType = Form(default=DocumentType.LAB_RESULT),
    file: UploadFile = File(...),
):
    logger.info(f"Upload: user={user_id} file={file.filename}")
    return await ingest(user_id=user_id, file=file, document_type=document_type)


@router.post("/manual")
async def create_manual_record(
    request: Request,
    body: UpdateResultRequest,
    user_id: str = Query(default=None, min_length=1, max_length=128),
):
    final_user_id = user_id or body.user_id
    if not final_user_id:
        raise HTTPException(status_code=422, detail="user_id is required in query or body")

    logger.info(f"Manual record request: URL={request.url} user_id={final_user_id}")
    document_id = str(uuid.uuid4())

    new_data = {
        "report_date": body.report_date,
        "lab_name": body.lab_name,
        "metrics": body.metrics
    }

    await _staging.save_manual(
        document_id=document_id,
        user_id=final_user_id,
        filename=body.lab_name or "Manual Entry",
        document_type=DocumentType.LAB_RESULT,
        structured_data=new_data
    )

    return {"message": "Manual record saved.", "document_id": document_id}


@router.get("/staging/{document_id}")
async def get_staging_preview(
    document_id: str,
    user_id: str = Query(..., min_length=1, max_length=128),
):
    record = await _staging.get_staging(document_id, user_id)
    if not record:
        raise HTTPException(
            status_code=404,
            detail=f"Staging record not found for document_id={document_id}.",
        )
    return record


@router.get("/staging")
async def list_staging(
    user_id: str = Query(..., min_length=1, max_length=128),
):
    records = await _staging.list_pending(user_id)
    return {"user_id": user_id, "pending": records, "count": len(records)}


@router.post("/confirm/{document_id}", response_model=ConfirmResponse)
async def confirm_document(
    document_id: str,
    user_id: str = Query(..., min_length=1, max_length=128),
    body: UpdateResultRequest | None = None,
):
    new_data = None
    if body:
        new_data = {
            "report_date": body.report_date,
            "lab_name": body.lab_name,
            "metrics": body.metrics
        }

    saved = await _staging.confirm(document_id=document_id, user_id=user_id, new_data=new_data)
    if not saved:
        raise HTTPException(status_code=404, detail="Staging record not found.")

    return ConfirmResponse(
        document_id=document_id,
        user_id=user_id,
        saved=True,
        message="Structured data saved to your health record.",
    )


@router.post("/discard/{document_id}", response_model=DiscardResponse)
async def discard_document(
    document_id: str,
    user_id: str = Query(..., min_length=1, max_length=128),
    remove_from_rag: bool = Query(default=True),
):
    discarded = await _staging.discard(document_id=document_id, user_id=user_id)
    if not discarded:
        raise HTTPException(status_code=404, detail="Staging record not found.")

    rag_chunks_removed = 0
    if remove_from_rag:
        try:
            rag_chunks_removed = await _rag_svc.delete_document(user_id, document_id)
        except Exception as exc:
            logger.error(f"RAG delete failed on discard: {exc}")

    return DiscardResponse(
        document_id=document_id,
        user_id=user_id,
        rag_chunks_removed=rag_chunks_removed,
        message="Document discarded."
    )


@router.get("/results")
async def get_lab_results(user_id: str = Query(...)):
    results = await _staging.get_lab_results(user_id)
    return {"user_id": user_id, "results": results, "count": len(results)}


@router.get("/documents")
async def list_documents(user_id: str = Query(...)):
    docs = await _vector_store.list_documents(user_id)
    return {"user_id": user_id, "documents": docs, "count": len(docs)}


@router.put("/results/{document_id}")
async def update_result(
    document_id: str,
    body: UpdateResultRequest,
    user_id: str = Query(...),
):
    new_data = {
        "report_date": body.report_date,
        "lab_name": body.lab_name,
        "metrics": body.metrics
    }

    updated = await _staging.update_result(document_id, user_id, new_data)
    if not updated:
        raise HTTPException(status_code=404, detail="Result not found.")

    return {"message": "Result updated.", "document_id": document_id}


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str, user_id: str = Query(...)):
    chunks_deleted = await _rag_svc.delete_document(user_id, document_id)
    db_deleted = await _staging.delete_result(document_id, user_id)

    if chunks_deleted == 0 and not db_deleted:
        raise HTTPException(status_code=404, detail="Document not found.")

    return {
        "message": "Document deleted.",
        "document_id": document_id,
        "database_deleted": db_deleted
    }


@router.get("/health")
async def health():
    return {"status": "ok", "service": "pipeline"}