"""
비동기 도우미 모듈

비동기 작업 처리를 위한 유틸리티 함수를 제공합니다.
"""

import asyncio
import functools
import logging
from typing import Any, Callable, Coroutine, TypeVar, ParamSpec, Awaitable

# 타입 변수 정의
T = TypeVar('T')
P = ParamSpec('P')

# 로거 설정
logger = logging.getLogger(__name__)


async def run_with_semaphore(semaphore: asyncio.Semaphore, coro: Coroutine[Any, Any, T]) -> T:
    """
    세마포어로 제한된 코루틴 실행

    Args:
        semaphore: 동시성 제한을 위한 세마포어
        coro: 실행할 코루틴

    Returns:
        실행 결과
    """
    async with semaphore:
        return await coro


async def run_with_timeout(coro: Coroutine[Any, Any, T], timeout: float) -> T:
    """
    제한 시간이 있는 코루틴 실행

    Args:
        coro: 실행할 코루틴
        timeout: 제한 시간(초)

    Returns:
        실행 결과

    Raises:
        asyncio.TimeoutError: 제한 시간 초과 시
    """
    return await asyncio.wait_for(coro, timeout=timeout)


def async_retry(max_retries: int = 3, backoff_factor: float = 1.5):
    """
    비동기 함수에 대한 재시도 데코레이터

    Args:
        max_retries: 최대 재시도 횟수
        backoff_factor: 재시도 간 대기 시간 증가 계수

    Returns:
        데코레이트된 함수
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    wait_time = backoff_factor ** attempt
                    logger.warning(
                        f"재시도 {attempt + 1}/{max_retries}: "
                        f"{func.__name__} - {wait_time:.2f}초 후 재시도"
                    )
                    await asyncio.sleep(wait_time)

            # 모든 재시도 실패
            logger.error(f"모든 재시도 실패: {func.__name__}")
            raise last_exception

        return wrapper

    return decorator
