from __future__ import annotations

import asyncio
import io
import json
from dataclasses import dataclass
from typing import Optional

from PIL import Image
from google import genai
from google.genai import types

from models.schemas import StructuredLabResult
from utils.logger import get_logger
from config import get_settings

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class ExtractionResult:
    raw_text: str  # Kept for compatibility with your Next.js/Golang pipeline
    structured: Optional[StructuredLabResult]
    confidence: float
    page_count: int = 1


class OCRService:
    """
    Async Multimodal LLM Extraction Service.
    
    Passes images directly to Gemini via the official google-genai SDK 
    to extract lab results into strict JSON. No local OCR required.
    """

    def __init__(self) -> None:
        # Initialize the new Google GenAI client.
        # Calling .aio gives us the fully asynchronous client for FastAPI.
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY).aio
        
        self.system_prompt = (
            "You are a medical data extraction assistant. Extract the lab parameters, "
            "patient info, and metadata from the provided document image(s). "
            "Return ONLY a valid JSON object matching the requested schema. "
            "If a value is not found, use null."
        )

    # ── Public API ────────────────────────────────────────────────

    async def process_image(self, image_bytes: bytes) -> ExtractionResult:
        """Process a single image directly through Gemini."""
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Cannot open image bytes: {exc}") from exc

        return await self._extract_via_gemini([img])

    async def process_pdf(self, pdf_bytes: bytes) -> ExtractionResult:
        """Rasterise each page, then pass all pages to Gemini at once."""
        loop = asyncio.get_event_loop()
        
        # Convert PDF to a list of PNG bytes in a background thread
        page_images_bytes: list[bytes] = await loop.run_in_executor(
            None, _pdf_to_images, pdf_bytes
        )
        
        if not page_images_bytes:
            return ExtractionResult(raw_text="", structured=None, confidence=0.0)

        # Convert bytes to PIL Images for Gemini
        pil_images = [
            Image.open(io.BytesIO(pb)).convert("RGB") for pb in page_images_bytes
        ]

        return await self._extract_via_gemini(pil_images)

    # ── Internal: Gemini Extraction ───────────────────────────────

    async def _extract_via_gemini(self, images: list[Image.Image]) -> ExtractionResult:
        """Passes the prompt and images to Gemini and parses the JSON."""
        
        # The new SDK accepts lists containing text strings and PIL Images natively
        contents = [self.system_prompt] + images

        try:
            # Call Gemini asynchronously using the new SDK
            response = await self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    # Force structured JSON by passing your Pydantic model directly
                    response_mime_type="application/json",
                    response_schema=StructuredLabResult, 
                )
            )
            
            # The response text is guaranteed to be a JSON string matching your schema
            json_text = response.text
            
            # Parse it strictly into your Pydantic schema
            parsed_data = StructuredLabResult.model_validate_json(json_text)
            
            return ExtractionResult(
                raw_text="Extracted directly via Multimodal LLM", # Fallback text
                structured=parsed_data,
                confidence=0.99, # LLMs don't give character confidence, so we mock it
                page_count=len(images)
            )
            
        except Exception as e:
            logger.error(f"Gemini Extraction failed: {e}")
            return ExtractionResult(
                raw_text="", 
                structured=None, 
                confidence=0.0,
                page_count=len(images)
            )


# ── Module-level PDF helper ───────────────────────────────────────────────────

def _pdf_to_images(pdf_bytes: bytes) -> list[bytes]:
    """Rasterise each PDF page to PNG bytes."""
    try:
        from pdf2image import convert_from_bytes
    except ImportError:
        raise RuntimeError("pdf2image not installed.")
        
    images = convert_from_bytes(pdf_bytes, dpi=200)
    result = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result.append(buf.getvalue())
    return result
