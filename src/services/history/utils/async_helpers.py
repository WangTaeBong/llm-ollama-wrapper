"""
비동기 도우미 모듈

비동기 작업 처리를 위한 유틸리티 함수를 제공합니다.
"""

import asyncio
import logging
import time
from typing import TypeVar, Any, Callable, Coroutine, Optional

# 타입 변수 정의
T = TypeVar('T')

# 로거 설정
logger = logging.getLogger(__name__)


async def run_with_retry(
        coro: Coroutine[Any, Any, T],
        max_retries: int = 3,
        retry_delay: float = 0.5,
        backoff_factor: float = 1.5,
        session_id: str = "unknown"
) -> T:
    """
    재시도 메커니즘이 포함된 코루틴 실행

    Args:
        coro: 실행할 코루틴
        max_retries: 최대 재시도 횟수
        retry_delay: 초기 재시도 지연 시간(초)
        backoff_factor: 재시도 간 대기 시간 증가 계수
        session_id: 로깅용 세션 ID

    Returns:
        T: 코루틴 실행 결과

    Raises:
        Exception: 모든 재시도가 실패한 경우 마지막 발생한 예외
    """
    retry_count = 0
    last_exception = None

    while retry_count <= max_retries:
        try:
            if retry_count > 0:
                logger.info(f"[{session_id}] 작업 재시도 {retry_count}/{max_retries}")

            return await coro

        except Exception as e:
            retry_count += 1
            last_exception = e

            if retry_count > max_retries:
                # 최대 재시도 횟수 초과
                logger.error(f"[{session_id}] 모든 재시도 실패: {str(e)}")
                break

            # 지수 백오프 적용
            wait_time = retry_delay * (backoff_factor ** (retry_count - 1))
            logger.warning(
                f"[{session_id}] 오류 발생, {wait_time:.2f}초 후 재시도 ({retry_count}/{max_retries}): {str(e)}"
            )

            # 재시도 전 대기
            await asyncio.sleep(wait_time)

    # 모든 재시도 실패
    raise last_exception


async def run_with_timeout(
        coro: Coroutine[Any, Any, T],
        timeout: float,
        session_id: str = "unknown"
) -> T:
    """
    제한 시간이 적용된 코루틴 실행

    제한 시간을 초과하면 TimeoutError를 발생시킵니다.

    Args:
        coro: 실행할 코루틴
        timeout: 제한 시간(초)
        session_id: 로깅용 세션 ID

    Returns:
        T: 코루틴 실행 결과

    Raises:
        asyncio.TimeoutError: 제한 시간 초과 시
    """
    start_time = time.time()

    try:
        result = await asyncio.wait_for(coro, timeout=timeout)
        elapsed = time.time() - start_time
        logger.debug(f"[{session_id}] 작업 완료: {elapsed:.4f}초 소요 (제한: {timeout}초)")
        return result

    except asyncio.TimeoutError as e:
        elapsed = time.time() - start_time
        logger.error(f"[{session_id}] 작업 시간 초과: {elapsed:.4f}초 소요 (제한: {timeout}초)")
        raise asyncio.TimeoutError(f"작업이 {timeout}초 제한 시간을 초과했습니다: {e}")

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[{session_id}] 작업 중 오류 발생: {elapsed:.4f}초 후 {str(e)}")
        raise


def fire_and_forget(
        coro: Coroutine[Any, Any, Any],
        session_id: str = "unknown",
        background_tasks: Optional[list] = None
) -> asyncio.Task:
    """
    결과를 기다리지 않고 코루틴을 비동기로 실행합니다.
    오류가 발생해도 주 실행 흐름에 영향을 주지 않습니다.

    Args:
        coro: 실행할 코루틴
        session_id: 로깅용 세션 ID
        background_tasks: 생성된 태스크를 추가할 리스트 (선택 사항)

    Returns:
        asyncio.Task: 생성된 비동기 태스크
    """

    async def wrapper():
        try:
            await coro
        except Exception as e:
            logger.error(f"[{session_id}] 백그라운드 태스크 오류: {str(e)}")

    task = asyncio.create_task(wrapper())

    # 생성된 태스크를 추적 리스트에 추가 (선택 사항)
    if background_tasks is not None:
        background_tasks.append(task)

        # 완료 시 리스트에서 제거하는 콜백 추가
        task.add_done_callback(
            lambda t: background_tasks.remove(t) if t in background_tasks else None
        )

    return task
