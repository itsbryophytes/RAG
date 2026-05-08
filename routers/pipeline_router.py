"""
routers/pipeline_router.py — Document ingestion + confirmation endpoints.

New endpoints:
  POST   /api/pipeline/upload                 → ingest (OCR + RAG + staging)
  GET    /api/pipeline/staging/{document_id}  → fetch staging preview
  GET    /api/pipeline/staging                → list pending staging records
  POST   /api/pipeline/confirm/{document_id}  → confirm → save to lab_results
  POST   /api/pipeline/discard/{document_id}  → discard → optionally remove from RAG
  GET    /api/pipeline/results                → list confirmed lab results (dashboard)
  GET    /api/pipeline/documents              → list RAG-indexed documents
  DELETE /api/pipeline/documents/{doc_id}     → delete from RAG

Old /api/pipeline/upload is now backed by ingestion_pipeline.ingest()
instead of document_pipeline.ingest_document().
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query

from models.schemas import IngestionResponse, ConfirmResponse, DiscardResponse
from models.enums import DocumentType
from pipelines.ingestion_pipeline import ingest
from services.rag_service import RAGService
from stores.staging_store import StagingStore
from stores.pgvector_store import PGVectorStore
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

_staging  = StagingStore()
_rag_svc  = RAGService()
_vector_store = PGVectorStore()


# ── Upload ────────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=IngestionResponse)
async def upload_document(
    user_id: str = Form(..., min_length=1, max_length=128),
    document_type: DocumentType = Form(default=DocumentType.LAB_RESULT),
    file: UploadFile = File(...),
):
    """
    Upload a lab result image or PDF.

    Immediately:
      - Runs OCR and parameter extraction
      - Indexes text into vector DB (RAG is ready for chat)
      - Saves structured data to staging for preview

    Returns a preview of extracted parameters.
    The user must call /confirm or /discard to finalise.

    Supported: JPEG, PNG, TIFF, BMP, WebP, PDF (max 20 MB)
    """
    logger.info(f"Upload: user={user_id} file={file.filename}")
    return await ingest(user_id=user_id, file=file, document_type=document_type)


# ── Staging preview ───────────────────────────────────────────────────────────

@router.get("/staging/{document_id}")
async def get_staging_preview(
    document_id: str,
    user_id: str = Query(..., min_length=1, max_length=128),
):
    """
    Retrieve the staging preview for a specific document.
    Used by the frontend to re-fetch the preview if needed.
    """
    record = await _staging.get_staging(document_id, user_id)
    if not record:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Staging record not found for document_id={document_id}. "
                "It may have expired (TTL 24h) or already been confirmed/discarded."
            ),
        )
    return record


@router.get("/staging")
async def list_staging(
    user_id: str = Query(..., min_length=1, max_length=128),
):
    """List all pending staging records (uploads awaiting confirmation)."""
    records = await _staging.list_pending(user_id)
    return {"user_id": user_id, "pending": records, "count": len(records)}


# ── Confirm ───────────────────────────────────────────────────────────────────

@router.post("/confirm/{document_id}", response_model=ConfirmResponse)
async def confirm_document(
    document_id: str,
    user_id: str = Query(..., min_length=1, max_length=128),
):
    """
    Confirm a document preview → saves structured data to the permanent
    lab_results table. RAG vectors are already in place from upload.

    This is the user's explicit consent to add this data to their health record.
    """
    saved = await _staging.confirm(document_id=document_id, user_id=user_id)
    if not saved:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No pending staging record found for document_id={document_id}. "
                "It may have expired (24h TTL), already been confirmed, or discarded."
            ),
        )
    return ConfirmResponse(
        document_id=document_id,
        user_id=user_id,
        saved=True,
        message="Structured data saved to your health record.",
    )


# ── Discard ───────────────────────────────────────────────────────────────────

@router.post("/discard/{document_id}", response_model=DiscardResponse)
async def discard_document(
    document_id: str,
    user_id: str = Query(..., min_length=1, max_length=128),
    remove_from_rag: bool = Query(
        default=True,
        description="If true, also removes the document's vectors from the chat index.",
    ),
):
    """
    Discard a document preview.
    Structured data is marked as discarded and never saved to lab_results.

    remove_from_rag (default: true):
      Set to false if you want the user to still chat about the document
      even after discarding it from their health record.
    """
    discarded = await _staging.discard(document_id=document_id, user_id=user_id)
    if not discarded:
        raise HTTPException(
            status_code=404,
            detail=f"No pending staging record found for document_id={document_id}.",
        )

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
        message=(
            "Document discarded. "
            + (
                f"{rag_chunks_removed} RAG chunks removed."
                if remove_from_rag
                else "RAG vectors kept (remove_from_rag=false)."
            )
        ),
    )


# ── Dashboard: confirmed results ──────────────────────────────────────────────

@router.get("/results")
async def get_lab_results(
    user_id: str = Query(..., min_length=1, max_length=128),
):
    """
    Return all confirmed lab results for the user's dashboard.
    Only records that the user explicitly confirmed are included.
    """
    results = await _staging.get_lab_results(user_id)
    return {"user_id": user_id, "results": results, "count": len(results)}


# ── RAG document management ───────────────────────────────────────────────────

@router.get("/documents")
async def list_documents(
    user_id: str = Query(..., min_length=1, max_length=128),
):
    """List all documents currently indexed in the vector store for this user."""
    docs = await _vector_store.list_documents(user_id)
    return {"user_id": user_id, "documents": docs, "count": len(docs)}


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    user_id: str = Query(..., min_length=1, max_length=128),
):
    """Remove a document's vectors from the RAG index."""
    deleted = await _rag_svc.delete_document(user_id, document_id)
    if deleted == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Document {document_id} not found in RAG index for user {user_id}.",
        )
    return {"message": f"Deleted {deleted} chunks.", "document_id": document_id}


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health")
async def health():
    return {"status": "ok", "service": "pipeline"}
