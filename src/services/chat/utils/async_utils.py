"""
비동기 유틸리티 모듈

채팅 서비스의 비동기 작업 처리를 위한 고급 유틸리티 함수를 제공합니다.
재시도 메커니즘, 타임아웃 처리, 동시성 제어, 백그라운드 작업 관리 등의 기능을 포함합니다.
"""

import asyncio
import functools
import logging
import time
from typing import TypeVar, Any, Callable, Coroutine, Optional, List, Dict

# 타입 변수 정의
T = TypeVar('T')  # 반환 타입
P = TypeVar('P')  # 매개변수 타입

# 로거 설정
logger = logging.getLogger(__name__)


async def run_with_retry(
        coro: Coroutine[Any, Any, T],
        max_retries: int = 3,
        retry_delay: float = 0.5,
        backoff_factor: float = 1.5,
        circuit_breaker: Optional[Any] = None,
        session_id: str = "unknown"
) -> T:
    """
    재시도 메커니즘이 포함된 코루틴 실행 함수

    지정된 코루틴을 실행하며, 실패 시 지수 백오프(exponential backoff) 전략으로 재시도합니다.
    서킷 브레이커와 통합되어 외부 서비스 보호 기능을 제공합니다.

    Args:
        coro: 실행할 코루틴
        max_retries: 최대 재시도 횟수 (기본값: 3)
        retry_delay: 초기 재시도 지연 시간(초) (기본값: 0.5)
        backoff_factor: 재시도 간 대기 시간 증가 계수 (기본값: 1.5)
        circuit_breaker: 선택적 서킷 브레이커 인스턴스
        session_id: 로깅용 세션 ID (기본값: "unknown")

    Returns:
        T: 코루틴 실행 결과

    Raises:
        Exception: 모든 재시도가 실패한 경우 마지막으로 발생한 예외
    """
    retry_count = 0
    last_exception = None

    while retry_count <= max_retries:
        try:
            # 서킷 브레이커 확인
            if circuit_breaker and circuit_breaker.is_open():
                logger.warning(f"[{session_id}] 서킷 열림, {coro.__name__} 호출 건너뜀")
                raise RuntimeError(f"서비스 사용 불가: {coro.__name__}에 대한 서킷이 열려 있습니다")

            # 재시도 카운트 로깅 (첫 시도가 아닌 경우)
            if retry_count > 0:
                logger.info(f"[{session_id}] 작업 재시도 {retry_count}/{max_retries}")

            # 코루틴 실행
            start_time = time.time()
            result = await coro
            execution_time = time.time() - start_time

            # 실행 시간 로깅
            logger.debug(f"[{session_id}] 함수 {coro.__name__} 완료: {execution_time:.4f}초")

            # 서킷 브레이커에 성공 기록
            if circuit_breaker:
                circuit_breaker.record_success()

            return result

        except (asyncio.TimeoutError, ConnectionError) as e:
            # 특정 예외는 재시도 가능한 것으로 처리
            if circuit_breaker:
                circuit_breaker.record_failure()

            retry_count += 1
            wait_time = retry_delay * (backoff_factor ** (retry_count - 1))
            last_exception = e

            if retry_count <= max_retries:
                logger.warning(
                    f"[{session_id}] 재시도 {retry_count}/{max_retries} - "
                    f"{wait_time:.2f}초 후 재시도 - 원인: {type(e).__name__}: {str(e)}"
                )
                # 재시도 전 대기
                await asyncio.sleep(wait_time)
            else:
                # 모든 재시도 실패
                logger.error(f"[{session_id}] 모든 {max_retries}회 재시도 실패")
                raise last_exception or RuntimeError(f"{coro.__name__}에 대한 모든 재시도 실패")

        except Exception as e:
            # 기타 예외는 서킷 브레이커에 실패 기록하고 바로 전파
            logger.error(f"[{session_id}] 예외 발생: {e}")
            if circuit_breaker:
                circuit_breaker.record_failure()
            raise

    # 모든 재시도 실패 (여기까지 오면 안 되지만 타입 체커를 위해 명시적으로 추가)
    raise last_exception or RuntimeError(f"{coro.__name__}에 대한 모든 재시도 실패")


def async_retry(max_retries: int = 3, backoff_factor: float = 1.5, circuit_breaker: Optional[Any] = None):
    """
    비동기 함수에 재시도 기능을 추가하는 데코레이터

    지수 백오프 전략으로 비동기 함수의 재시도를 구현하며, 서킷 브레이커와 통합됩니다.
    함수 수준에서 재시도 전략을 적용할 때 편리합니다.

    Args:
        max_retries: 최대 재시도 횟수 (기본값: 3)
        backoff_factor: 재시도 간 대기 시간 증가 계수 (기본값: 1.5)
        circuit_breaker: 선택적 서킷 브레이커 인스턴스

    Returns:
        Callable: 재시도 로직이 포함된 데코레이터 함수
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # 세션 ID를 키워드 인수에서 추출하거나 기본값 사용
            session_id = kwargs.get('session_id', 'unknown')
            if 'session_id' not in kwargs and args and hasattr(args[0], 'current_session_id'):
                session_id = args[0].current_session_id

            retry_count = 0
            last_exception = None

            while retry_count < max_retries:
                # 서킷 브레이커 확인
                if circuit_breaker and circuit_breaker.is_open():
                    logger.warning(f"[{session_id}] 서킷 열림, {func.__name__} 호출 건너뜀")
                    raise RuntimeError(f"서비스 사용 불가: {func.__name__}에 대한 서킷이 열려 있습니다")

                try:
                    start_time = time.time()
                    result = await func(*args, **kwargs)
                    execution_time = time.time() - start_time

                    # 실행 시간 로깅
                    logger.debug(f"[{session_id}] 함수 {func.__name__} 완료: {execution_time:.4f}초")

                    # 서킷 브레이커에 성공 기록
                    if circuit_breaker:
                        circuit_breaker.record_success()

                    return result

                except (asyncio.TimeoutError, ConnectionError) as e:
                    # 특정 예외는 재시도 가능한 것으로 처리
                    if circuit_breaker:
                        circuit_breaker.record_failure()

                    retry_count += 1
                    wait_time = backoff_factor ** retry_count
                    last_exception = e

                    if retry_count < max_retries:
                        logger.warning(
                            f"[{session_id}] 재시도 {retry_count}/{max_retries} - "
                            f"{func.__name__} {wait_time:.2f}초 후 재시도 - 원인: {type(e).__name__}: {str(e)}"
                        )
                        # 재시도 전 대기
                        await asyncio.sleep(wait_time)
                    else:
                        # 모든 재시도 실패
                        logger.error(f"[{session_id}] 모든 {max_retries}회 재시도 실패: {func.__name__}")
                        raise

                except Exception as e:
                    logger.error(f"[{session_id}] async_retry 예외: {e}")
                    # 기타 예외는 서킷 브레이커에 실패 기록하고 바로 전파
                    if circuit_breaker:
                        circuit_breaker.record_failure()
                    raise

            # 모든 재시도 실패
            raise last_exception or RuntimeError(f"모든 재시도 실패: {func.__name__}")

        return wrapper
    return decorator


async def run_with_timeout(
        coro: Coroutine[Any, Any, T],
        timeout: float,
        session_id: str = "unknown"
) -> T:
    """
    제한 시간이 적용된 코루틴 실행 함수

    지정된 시간 내에 코루틴이 완료되지 않으면 TimeoutError를 발생시킵니다.
    실행 시간을 측정하고 로깅하는 기능을 포함합니다.

    Args:
        coro: 실행할 코루틴
        timeout: 제한 시간(초)
        session_id: 로깅용 세션 ID (기본값: "unknown")

    Returns:
        T: 코루틴 실행 결과

    Raises:
        asyncio.TimeoutError: 제한 시간 초과 시
        Exception: 코루틴 실행 중 발생한 기타 예외
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


async def run_with_semaphore(semaphore: asyncio.Semaphore, coro: Coroutine[Any, Any, T]) -> T:
    """
    세마포어로 제한된 코루틴 실행 함수

    동시성을 제어하기 위해 세마포어를 사용하여 코루틴을 실행합니다.

    Args:
        semaphore: 동시성 제어용 세마포어
        coro: 실행할 코루틴

    Returns:
        T: 코루틴 실행 결과
    """
    async with semaphore:
        return await coro


def fire_and_forget(
        coro: Coroutine[Any, Any, Any],
        session_id: str = "unknown",
        background_tasks: Optional[List[asyncio.Task]] = None
) -> asyncio.Task:
    """
    결과를 기다리지 않고 코루틴을 비동기로 실행합니다.

    백그라운드 작업으로 코루틴을 실행하며, 오류 처리와 작업 추적 기능을 제공합니다.
    주 실행 흐름에 영향을 주지 않고 비동기 작업을 실행할 때 유용합니다.

    Args:
        coro: 실행할 코루틴
        session_id: 로깅용 세션 ID (기본값: "unknown")
        background_tasks: 생성된 태스크를 추가할 리스트 (선택 사항)

    Returns:
        asyncio.Task: 생성된 비동기 태스크
    """
    async def wrapper():
        try:
            await coro
        except Exception as e:
            logger.error(f"[{session_id}] 백그라운드 태스크 오류: {str(e)}", exc_info=True)

    task = asyncio.create_task(wrapper())

    # 생성된 태스크를 추적 리스트에 추가 (선택 사항)
    if background_tasks is not None:
        background_tasks.append(task)

        # 완료 시 리스트에서 제거하는 콜백 추가
        task.add_done_callback(
            lambda t: background_tasks.remove(t) if t in background_tasks else None
        )

    return task


class AsyncTaskManager:
    """
    비동기 작업 관리 클래스

    백그라운드 작업들을 추적하고 관리하는 기능을 제공합니다.
    작업 생성, 취소, 상태 확인, 로깅 등의 기능을 포함합니다.
    애플리케이션 종료 시 진행 중인 작업을 안전하게 처리하기 위해 사용될 수 있습니다.
    """

    def __init__(self):
        """비동기 작업 관리자 초기화"""
        self._tasks: List[asyncio.Task] = []
        self._task_metadata: Dict[asyncio.Task, Dict[str, Any]] = {}

    def create_task(self, coro: Coroutine, name: str = None, session_id: str = "unknown") -> asyncio.Task:
        """
        비동기 작업을 생성하고 추적합니다.

        Args:
            coro: 실행할 코루틴
            name: 작업 이름 (선택 사항)
            session_id: 세션 ID (기본값: "unknown")

        Returns:
            asyncio.Task: 생성된 비동기 태스크
        """
        async def wrapper():
            try:
                return await coro
            except asyncio.CancelledError:
                logger.debug(f"[{session_id}] 작업 취소됨: {name or 'unnamed'}")
                raise
            except Exception as e:
                logger.error(f"[{session_id}] 작업 오류: {name or 'unnamed'} - {str(e)}", exc_info=True)
                raise

        task = asyncio.create_task(wrapper())

        # 메타데이터 설정 및 작업 추적
        metadata = {
            "name": name or "unnamed_task",
            "session_id": session_id,
            "start_time": time.time(),
            "status": "running"
        }
        self._task_metadata[task] = metadata
        self._tasks.append(task)

        # 완료 콜백 추가
        task.add_done_callback(self._task_done_callback)

        return task

    def _task_done_callback(self, task: asyncio.Task) -> None:
        """
        작업 완료 시 호출되는 콜백 함수

        Args:
            task: 완료된 작업
        """
        # 작업 목록에서 제거
        if task in self._tasks:
            self._tasks.remove(task)

        # 작업 메타데이터 업데이트 및 로깅
        if task in self._task_metadata:
            metadata = self._task_metadata[task]
            session_id = metadata.get("session_id", "unknown")
            name = metadata.get("name", "unnamed_task")
            start_time = metadata.get("start_time", time.time())
            elapsed = time.time() - start_time

            if task.cancelled():
                metadata["status"] = "cancelled"
                logger.debug(f"[{session_id}] 작업 취소됨: {name} ({elapsed:.4f}초)")
            elif task.exception():
                metadata["status"] = "failed"
                exception = task.exception()
                logger.error(f"[{session_id}] 작업 실패: {name} - {str(exception)} ({elapsed:.4f}초)")
            else:
                metadata["status"] = "completed"
                logger.debug(f"[{session_id}] 작업 완료: {name} ({elapsed:.4f}초)")

            metadata["end_time"] = time.time()
            metadata["elapsed"] = elapsed

    async def cancel_all_tasks(self, wait_for_completion: bool = True) -> None:
        """
        모든 진행 중인 작업을 취소합니다.

        Args:
            wait_for_completion: 취소 완료를 기다릴지 여부 (기본값: True)
        """
        if not self._tasks:
            return

        # 모든 작업 취소
        for task in self._tasks:
            if not task.done():
                task.cancel()

        # 취소 완료 대기
        if wait_for_completion and self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        # 작업 목록 비우기
        self._tasks.clear()

        logger.info(f"모든 작업이 취소되었습니다.")

    def get_running_tasks_info(self) -> List[Dict[str, Any]]:
        """
        실행 중인 작업 정보를 반환합니다.

        Returns:
            List[Dict[str, Any]]: 실행 중인 작업 정보 목록
        """
        running_tasks = []
        current_time = time.time()

        for task in self._tasks:
            if not task.done() and task in self._task_metadata:
                metadata = self._task_metadata[task]
                start_time = metadata.get("start_time", current_time)

                task_info = {
                    "name": metadata.get("name", "unnamed_task"),
                    "session_id": metadata.get("session_id", "unknown"),
                    "runtime": current_time - start_time,
                    "status": metadata.get("status", "unknown")
                }
                running_tasks.append(task_info)

        return running_tasks

    @property
    def active_task_count(self) -> int:
        """
        현재 활성 작업 수를 반환합니다.

        Returns:
            int: 활성 작업 수
        """
        return len([t for t in self._tasks if not t.done()])


# 전역 작업 관리자 인스턴스
task_manager = AsyncTaskManager()
