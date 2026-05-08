from __future__ import annotations

import uuid

from fastapi import UploadFile, HTTPException

from models.schemas import IngestionResponse
from models.enums import DocumentType
from services.ocr_service import OCRService
from services.rag_service import RAGService
from stores.staging_store import StagingStore
from utils.logger import get_logger

logger = get_logger(__name__)

_ocr_svc     = OCRService()
_rag_svc     = RAGService()
_staging     = StagingStore()

ALLOWED_MIME_TYPES = {
    "image/jpeg", "image/png", "image/tiff",
    "image/bmp",  "image/webp",
    "application/pdf",
}
MAX_FILE_SIZE_MB = 20


async def ingest(
    user_id: str,
    file: UploadFile,
    document_type: DocumentType = DocumentType.LAB_RESULT,
) -> IngestionResponse:

    content_type = (file.content_type or "").lower().split(";")[0].strip()
    if content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type: '{content_type}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_MIME_TYPES))}"
            ),
        )

    file_bytes = await file.read()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.1f} MB (max {MAX_FILE_SIZE_MB} MB).",
        )

    filename    = file.filename or "upload"
    document_id = str(uuid.uuid4())

    logger.info(
        f"Ingest start: user={user_id} file={filename} "
        f"type={content_type} size={size_mb:.2f}MB doc={document_id}"
    )

    try:
        if content_type == "application/pdf":
            extraction_result = await _ocr_svc.process_pdf(file_bytes)
        else:
            extraction_result = await _ocr_svc.process_image(file_bytes)
    except Exception as exc:
        logger.error(f"Extraction failed for document={document_id}: {exc}")
        raise HTTPException(status_code=422, detail=f"Document processing failed: {exc}")

    if not extraction_result.structured:
        raise HTTPException(
            status_code=422,
            detail=(
                "Failed to extract structured data from the document. "
                "Please ensure the image is clear and contains a valid lab report."
            ),
        )

    param_count = len(extraction_result.structured.parameters)
    logger.info(
        f"Extraction done: conf={extraction_result.confidence:.2f}, "
        f"params={param_count}, pages={extraction_result.page_count}"
    )

    param_summary_lines = [
        f"{p.name}: {p.value} {p.unit}" + (f" [{p.flag}]" if p.flag else "")
        for p in extraction_result.structured.parameters
    ]
    
    rag_text = (
        extraction_result.raw_text
        + "\n\nExtracted Parameters:\n"
        + "\n".join(param_summary_lines)
    )

    chunks_indexed = 0
    rag_ready = False
    try:
        chunks_indexed = await _rag_svc.index_text(
            user_id=user_id,
            document_id=document_id,
            text=rag_text,
            extra_metadata={
                "source_filename": filename,
                "document_type": document_type.value,
                "extraction_confidence": round(extraction_result.confidence, 3),
                "patient_name": extraction_result.structured.patient_name,
                "lab_name": extraction_result.structured.lab_name,
                "date": extraction_result.structured.date,
            },
        )
        rag_ready = chunks_indexed > 0
        logger.info(f"RAG indexed: {chunks_indexed} chunks for document={document_id}")
    except Exception as exc:
        logger.error(f"RAG indexing failed for document={document_id}: {exc}")

    structured_dict = _serialize_structured(extraction_result.structured)
    try:
        await _staging.save_staging(
            document_id=document_id,
            user_id=user_id,
            filename=filename,
            document_type=document_type,
            structured_data=structured_dict,
            ocr_confidence=extraction_result.confidence,
        )
    except Exception as exc:
        logger.error(f"Staging save failed for document={document_id}: {exc}")

    return IngestionResponse(
        document_id=document_id,
        user_id=user_id,
        filename=filename,
        document_type=document_type,
        chunks_indexed=chunks_indexed,
        ocr_confidence=round(extraction_result.confidence, 3),
        preview=structured_dict,
        rag_ready=rag_ready,
    )

def _serialize_structured(structured) -> dict | None:
    if not structured:
        return None
    return {
        "parameters": {
            p.name: {
                "value":        p.value,
                "unit":         p.unit,
                "raw_value":    p.raw_value,
                "flag":         p.flag,
                "normal_range": p.normal_range,
            }
            for p in structured.parameters
        },
        "lab_name":     structured.lab_name,
        "patient_name": structured.patient_name,
        "date":         structured.date,
        "confidence":   structured.confidence,
        "warnings":     structured.warnings,
    }