from __future__ import annotations

import io
import uuid
from pathlib import Path

from fastapi import UploadFile, HTTPException

from models.schemas import DocumentUploadResponse
from models.enums import DocumentType
from services.ocr_service import OCRService
from services.embedding_service import EmbeddingService
from stores.pgvector_store import PGVectorStore
from processing.chunker import chunk_text
from utils.logger import get_logger
from config import get_settings

logger = get_logger(__name__)
settings = get_settings()

_ocr_svc = OCRService()
_embedding_svc = EmbeddingService()
_vector_store = PGVectorStore()

ALLOWED_MIME_TYPES = {
    "image/jpeg", "image/png", "image/tiff", "image/bmp", "image/webp",
    "application/pdf",
}
MAX_FILE_SIZE_MB = 20


async def ingest_document(
    user_id: str,
    file: UploadFile,
    document_type: DocumentType = DocumentType.LAB_RESULT,
) -> DocumentUploadResponse:

    content_type = file.content_type or ""
    if content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {content_type}. "
                   f"Allowed: {', '.join(ALLOWED_MIME_TYPES)}",
        )

    file_bytes = await file.read()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.1f} MB (max {MAX_FILE_SIZE_MB} MB).",
        )

    filename = file.filename or "upload"
    document_id = str(uuid.uuid4())

    logger.info(
        f"Ingesting document: user={user_id}, "
        f"file={filename}, size={size_mb:.2f}MB, type={content_type}"
    )

    try:
        if content_type == "application/pdf":
            page_images = _pdf_to_images(file_bytes)
            clean_text, structured, confidence = await _ocr_svc.process_pdf_pages(page_images)
        else:
            clean_text, structured, confidence = await _ocr_svc.process_image(
                file_bytes, filename=filename
            )
    except Exception as exc:
        logger.error(f"OCR failed for document={document_id}: {exc}")
        raise HTTPException(status_code=422, detail=f"OCR processing failed: {exc}")

    if not clean_text.strip():
        raise HTTPException(
            status_code=422,
            detail="No text could be extracted from the document. "
                   "Please upload a clearer image.",
        )

    logger.info(
        f"OCR complete: {len(clean_text)} chars, "
        f"confidence={confidence:.2f}, "
        f"structured_params={len(structured.parameters) if structured else 0}"
    )

    embed_text = clean_text
    if structured and structured.parameters:
        param_lines = []
        for name, param in structured.parameters.items():
            flag_str = f" [{param.flag}]" if param.flag else ""
            param_lines.append(
                f"{name}: {param.value} {param.unit}{flag_str}"
            )
        embed_text = clean_text + "\n\nExtracted Parameters:\n" + "\n".join(param_lines)

    extra_meta = {
        "source_filename": filename,
        "document_type": document_type.value,
        "ocr_confidence": round(confidence, 3),
    }
    chunks = chunk_text(
        text=embed_text,
        document_id=document_id,
        extra_metadata=extra_meta,
    )

    if not chunks:
        raise HTTPException(
            status_code=422, detail="Document produced no embeddable chunks."
        )

    logger.info(f"Created {len(chunks)} chunks for document={document_id}")

    try:
        texts = [c.content for c in chunks]
        embeddings = await _embedding_svc.embed_documents_batch(texts)
    except Exception as exc:
        logger.error(f"Embedding batch failed: {exc}")
        raise HTTPException(status_code=502, detail=f"Embedding service error: {exc}")

    chunk_records = [
        {
            "chunk_id": chunk.chunk_id,
            "content": chunk.content,
            "embedding": embeddings[i],
            "metadata": chunk.metadata,
        }
        for i, chunk in enumerate(chunks)
    ]

    await _vector_store.upsert_chunks_bulk(user_id, document_id, chunk_records)

    return DocumentUploadResponse(
        document_id=document_id,
        user_id=user_id,
        filename=filename,
        document_type=document_type,
        chunks_created=len(chunks),
        structured_data=_serialize_structured(structured),
        ocr_confidence=round(confidence, 3),
    )


def _pdf_to_images(pdf_bytes: bytes) -> list[bytes]:
    try:
        from pdf2image import convert_from_bytes
    except ImportError:
        raise RuntimeError(
            "pdf2image is required for PDF processing. "
            "Install with: pip install pdf2image && apt install poppler-utils"
        )

    pil_images = convert_from_bytes(pdf_bytes, dpi=200)
    page_bytes = []
    for img in pil_images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        page_bytes.append(buf.getvalue())
    return page_bytes


def _serialize_structured(structured) -> dict | None:
    if not structured:
        return None
    return {
        "parameters": {
            name: {
                "value": param.value,
                "unit": param.unit,
                "flag": param.flag,
                "normal_range": param.normal_range,
            }
            for name, param in structured.parameters.items()
        },
        "lab_name": structured.lab_name,
        "date": structured.date,
        "warnings": structured.warnings,
    }