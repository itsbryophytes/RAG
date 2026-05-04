import asyncio
from typing import AsyncIterator
from functools import lru_cache

from google import genai
from google.genai.types import GenerateContentConfig, HarmCategory, HarmBlockThreshold

from config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

@lru_cache(maxsize=1)
def get_client():
    return genai.Client(api_key=settings.GEMINI_API_KEY)

SAFETY_SETTINGS = [
    {
        "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
        "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    },
    {
        "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    },
    {
        "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    },
    {
        "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH,
    },
]

GENERATION_CONFIG = GenerateContentConfig(
    temperature=0.1,
    top_p=0.8,
    top_k=40,
    max_output_tokens=2048,
    safety_settings=SAFETY_SETTINGS,
)

class GeminiService:
    def __init__(self) -> None:
        self.client = get_client()
        self.model = settings.GEMINI_CHAT_MODEL

    async def stream_chat(
        self,
        messages: list[dict],
        system_instruction: str,
    ) -> AsyncIterator[str]:

        loop = asyncio.get_event_loop()

        try:
            full_prompt = self._build_prompt(messages, system_instruction)

            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content_stream(
                    model=self.model,
                    contents=full_prompt,
                    config=GENERATION_CONFIG,
                ),
            )

            for chunk in response:
                text = getattr(chunk, "text", None)
                if text:
                    yield text

        except Exception as exc:
            logger.error(f"Gemini streaming error: {exc}")
            yield "[Error generating response. Please try again.]"

    def _build_prompt(self, messages: list[dict], system_instruction: str) -> str:
        parts = []

        if system_instruction:
            parts.append(system_instruction)

        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")

        parts.append("\nAssistant:")
        return "\n".join(parts)

    async def complete(self, prompt: str) -> str:
        loop = asyncio.get_event_loop()

        response = await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=GENERATION_CONFIG,
            ),
        )
        return response.text