from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from models.schemas import ChatRequest
from pipelines.chat_pipeline import chat_stream
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/")
async def chat_endpoint(request: ChatRequest, http_request: Request):
    logger.info(
        f"Chat SSE: user={request.user_id} "
        f"client={http_request.client.host if http_request.client else 'unknown'}"
    )

    async def event_generator():
        try:
            async for chunk in chat_stream(request):
                if await http_request.is_disconnected():
                    logger.info(f"Client disconnected: user={request.user_id}")
                    break
                yield chunk
        except Exception as exc:
            import json
            logger.error(f"SSE generator error: {exc}")
            yield f"data: {json.dumps({'type': 'error', 'text': str(exc)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )

@router.get("/health")
async def health():
    return {"status": "ok", "service": "chat"}