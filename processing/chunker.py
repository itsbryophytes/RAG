import re
import uuid
from dataclasses import dataclass
from config import get_settings

settings = get_settings()

@dataclass
class Chunk:
    chunk_id: str
    content: str
    char_start: int
    char_end: int
    chunk_index: int
    metadata: dict


def _split_sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+|\n{2,}", text.strip())


def chunk_text(
    text: str,
    document_id: str,
    extra_metadata: dict | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
    meta = extra_metadata or {}

    if not text.strip():
        return []

    sentences = _split_sentences(text)
    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0
    chunk_index = 0
    char_cursor = 0

    def flush(sentences_in_chunk: list[str]) -> None:
        nonlocal chunk_index, char_cursor
        
        context_parts = []
        if meta.get("patient_name"):
            context_parts.append(f"Patient: {meta['patient_name']}")
        if meta.get("document_type"):
            context_parts.append(f"Type: {meta['document_type']}")
        if meta.get("lab_name"):
            context_parts.append(f"Lab: {meta['lab_name']}")
        if meta.get("date"):
            context_parts.append(f"Date: {meta['date']}")
            
        context_prefix = f"[{' | '.join(context_parts)}]\n" if context_parts else ""
        
        raw_content = " ".join(sentences_in_chunk).strip()
        if not raw_content:
            return
            
        content = context_prefix + raw_content
        start = char_cursor
        end = start + len(content)
        chunks.append(
            Chunk(
                chunk_id=str(uuid.uuid4()),
                content=content,
                char_start=start,
                char_end=end,
                chunk_index=chunk_index,
                metadata={
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    **meta,
                },
            )
        )
        chunk_index += 1
        char_cursor = max(0, end - chunk_overlap)

    for sentence in sentences:
        s_len = len(sentence)
        if current_len + s_len > chunk_size and current:
            flush(current)
            while current and current_len > chunk_overlap:
                removed = current.pop(0)
                current_len -= len(removed) + 1
        current.append(sentence)
        current_len += s_len + 1

    if current:
        flush(current)

    return chunks