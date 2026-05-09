from __future__ import annotations

from typing import Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from models.enums import DocumentType, ResponseSafety, StagingStatus

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str

class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1, max_length=4096)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)
    top_k: int = Field(default=6, ge=1, le=20)
    threshold: float = Field(default=0.65, ge=0.0, le=1.0)

class RetrievedChunk(BaseModel):
    chunk_id: str
    content: str
    score: float
    source_document_id: str
    metadata: dict[str, Any] = {}

class ChatResponseMeta(BaseModel):
    safety: ResponseSafety
    retrieved_chunks: int
    is_emergency: bool
    disclaimer: str = (
        "⚠️ This information is for educational purposes only. "
        "It is NOT medical advice. Always consult a qualified healthcare professional."
    )

class DocumentUploadResponse(BaseModel):
    document_id: str
    user_id: str
    filename: str
    document_type: DocumentType
    chunks_created: int
    structured_data: Optional[dict[str, Any]] = None
    ocr_confidence: Optional[float] = None
    message: str = "Document processed successfully."

class LabParameter(BaseModel):
    name: str = Field(description="The name of the parameter")
    value: float | None = None
    unit: str | None = None
    raw_value: str | None = None
    normal_range: str | None = None
    flag: str | None = None

class StructuredLabResult(BaseModel):
    parameters: list[LabParameter]
    raw_text: str | None = None
    confidence: float | None = None
    lab_name: str | None = None
    patient_name: str | None = None
    date: str | None = None
    warnings: list[str] = []

class IngestionResponse(BaseModel):
    document_id: str
    user_id: str
    filename: str
    document_type: DocumentType
    chunks_indexed: int
    ocr_confidence: float
    preview: Optional[dict[str, Any]] = None
    rag_ready: bool = True
    message: str = (
        "Document processed. RAG is ready."
        "Review the preview and confirm to save your health record."
    )

class StagingRecord(BaseModel):
    document_id: str
    user_id: str
    filename: str
    document_type: DocumentType
    structured_data: Optional[dict[str, Any]]
    ocr_confidence: float
    status: StagingStatus
    created_at: datetime
    expires_at: datetime

class ConfirmResponse(BaseModel):
    document_id: str
    user_id: str
    saved: bool
    message: str

class DiscardResponse(BaseModel):
    document_id: str
    user_id: str
    rag_chunks_removed: int
    message: str

class UpdateResultRequest(BaseModel):
    report_date: Optional[str] = None
    lab_name: Optional[str] = None
    metrics: dict[str, Any]
    user_id: Optional[str] = None