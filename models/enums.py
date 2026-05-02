from enum import Enum

class DocumentType(str, Enum):
    LAB_RESULT = "lab_result"
    PRESCRIPTION = "prescription"
    CLINICAL_NOTE = "clinical_note"
    OTHER = "other"

class ChunkStatus(str, Enum):
    PENDING = "pending"
    EMBEDDED = "embedded"
    FAILED = "failed"

class ResponseSafety(str, Enum):
    SAFE = "safe"
    EMERGENCY = "emergency"
    UNCERTAIN = "uncertain"