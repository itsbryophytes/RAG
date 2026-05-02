from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field
from models.enums import DocumentType, ResponseSafety


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
    value: float
    unit: str
    raw_value: str
    normal_range: Optional[str] = None
    flag: Optional[str] = None

class StructuredLabResult(BaseModel):
    parameters: dict[str, LabParameter]
    raw_text: str
    confidence: float
    lab_name: Optional[str] = None
    patient_name: Optional[str] = None
    date: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)