import asyncio
import functools
from typing import Callable, Type, Tuple
from utils.logger import get_logger

logger = get_logger(__name__)


def async_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):

    def decorator(fn: Callable):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_attempts:
                        logger.error(
                            f"{fn.__name__} failed after {max_attempts} attempts: {exc}"
                        )
                        raise
                    sleep = min(delay * (2 ** (attempt - 1)), max_delay)
                    logger.warning(
                        f"{fn.__name__} attempt {attempt}/{max_attempts} failed "
                        f"({exc}). Retrying in {sleep:.1f}s …"
                    )
                    await asyncio.sleep(sleep)
        return wrapper
    return decorator
