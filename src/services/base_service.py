"""
기본 서비스 모듈

모든 서비스 클래스의 기본이 되는 추상 클래스와 공통 기능을 제공합니다.
"""

import abc
import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Callable, Awaitable

# 로거 설정
logger = logging.getLogger(__name__)


class BaseService(abc.ABC):
    """
    모든 서비스의 기본 클래스

    로깅, 성능 측정, 기본 오류 처리 및 기타 공통 기능을 제공합니다.
    """

    # 클래스 레벨 로깅 큐
    _log_queue = asyncio.Queue()
    _log_task = None
    _log_initialized = False

    def __init__(self, request_id: str, settings: Any):
        """
        기본 서비스 초기화

        Args:
            request_id: 요청 식별자(세션 ID 등)
            settings: 설정 객체
        """
        self.request_id = request_id
        self.settings = settings

        # 성능 측정용 타이머
        self.start_time = time.time()
        self.processing_stages = {}

        # 성능 지표
        self.metrics = {
            "request_count": 0,
            "total_time": 0,
            "error_count": 0
        }

    async def _ensure_log_task_running(self):
        """
        비동기 로깅 태스크가 실행 중인지 확인하고, 필요하면 시작합니다.
        """
        if not BaseService._log_initialized:
            BaseService._log_task = asyncio.create_task(self._process_logs())
            BaseService._log_initialized = True
            logger.info("비동기 로깅 시스템이 초기화되었습니다")

    @classmethod
    async def _process_logs(cls):
        """
        로그 큐에서 로그 항목을 비동기적으로 처리합니다.

        이 메서드는 무한 루프에서 실행되며, 큐에서 로그 항목을 가져와
        적절한 로거 메서드로 전달합니다. 예외는 시스템 중단 없이 처리합니다.
        """
        while True:
            try:
                # 큐에서 로그 항목 가져오기
                log_entry = await cls._log_queue.get()
                level, message, extra = log_entry

                # 'exc_info' 파라미터 추출 및 extra에서 제거
                exc_info = extra.pop('exc_info', False) if isinstance(extra, dict) else False

                # 적절한 레벨로 로깅
                if level == "debug":
                    logger.debug(message, exc_info=exc_info, extra=extra)
                elif level == "info":
                    logger.info(message, exc_info=exc_info, extra=extra)
                elif level == "warning":
                    logger.warning(message, exc_info=exc_info, extra=extra)
                elif level == "error":
                    logger.error(message, exc_info=exc_info, extra=extra)

                # 태스크 완료 표시
                cls._log_queue.task_done()
            except Exception as e:
                print(f"로그 처리 오류: {e}")
                await asyncio.sleep(1)  # 타이트 루프 방지를 위한 대기

    async def log(self, level: str, message: str, **kwargs):
        """
        비동기 로깅 큐에 로그 항목을 추가합니다.

        Args:
            level: 로그 레벨 ("debug", "info", "warning", "error")
            message: 로그 메시지
            **kwargs: 로거에 전달할 추가 파라미터
        """
        # 로깅 태스크가 실행 중인지 확인
        await self._ensure_log_task_running()

        # 모든 로그에 request_id 추가
        if 'session_id' not in kwargs:
            kwargs['session_id'] = self.request_id

        await BaseService._log_queue.put((level, message, kwargs))

    def record_stage(self, stage_name: str) -> float:
        """
        처리 단계의 타이밍을 기록하고 경과 시간을 반환합니다.

        이 메서드는 마지막 단계 이후 경과된 시간을 계산하고,
        processing_stages 딕셔너리에 기록한 후, 다음 단계를 위해
        start_time을 재설정합니다.

        Args:
            stage_name: 처리 단계 이름

        Returns:
            float: 마지막으로 기록된 단계 이후 경과 시간(초)
        """
        current_time = time.time()
        elapsed = current_time - self.start_time
        self.processing_stages[stage_name] = elapsed
        self.start_time = current_time
        return elapsed

    def get_metrics(self) -> Dict[str, Any]:
        """
        서비스 성능 지표를 반환합니다.

        Returns:
            Dict[str, Any]: 서비스 지표
        """
        avg_time = 0
        if self.metrics["request_count"] > 0:
            avg_time = self.metrics["total_time"] / self.metrics["request_count"]

        return {
            "request_count": self.metrics["request_count"],
            "error_count": self.metrics["error_count"],
            "avg_response_time": avg_time,
            "total_time": self.metrics["total_time"],
            "processing_stages": self.processing_stages
        }

    def update_metrics(self, start_time: float, success: bool = True):
        """
        요청 처리 후 성능 지표를 업데이트합니다.

        Args:
            start_time: 요청 시작 시간
            success: 요청 성공 여부
        """
        elapsed = time.time() - start_time
        self.metrics["request_count"] += 1
        self.metrics["total_time"] += elapsed

        if not success:
            self.metrics["error_count"] += 1

    @staticmethod
    def is_gemma_model(settings: Any) -> bool:
        """
        현재 모델이 Gemma 모델인지 확인합니다.

        Args:
            settings: 설정 객체

        Returns:
            bool: Gemma 모델이면 True, 아니면 False
        """
        # LLM 백엔드 확인
        backend = getattr(settings.llm, 'llm_backend', '').lower()

        # OLLAMA 백엔드인 경우
        if backend == 'ollama':
            if hasattr(settings.ollama, 'model_name'):
                model_name = settings.ollama.model_name.lower()
                return 'gemma' in model_name

        # VLLM 백엔드인 경우
        elif backend == 'vllm':
            if hasattr(settings.llm, 'model_type'):
                model_type = settings.llm.model_type.lower() if hasattr(settings.llm.model_type, 'lower') else str(
                    settings.llm.model_type).lower()
                return model_type == 'gemma'

        # 기본적으로 False 반환
        return False

    @staticmethod
    def fire_and_forget(coro: Awaitable, background_tasks: Optional[List] = None):
        """
        코루틴을 비동기로 실행하고 결과를 기다리지 않습니다.

        백그라운드 태스크 추적 및 오류 처리 기능 포함.

        Args:
            coro: 실행할 코루틴
            background_tasks: 태스크를 추가할 배경 작업 목록(선택 사항)
        """

        async def wrapper():
            try:
                await coro
            except Exception as e:
                logger.error(f"백그라운드 태스크 오류: {e}", exc_info=True)

        task = asyncio.create_task(wrapper())

        # 제공된 경우 배경 작업 목록에 추가
        if background_tasks is not None:
            background_tasks.append(task)

            # 완료 시 목록에서 제거하는 콜백 추가
            task.add_done_callback(
                lambda t: background_tasks.remove(t) if t in background_tasks else None
            )

        return task

    async def handle_request_with_retry(
            self,
            func: Callable[..., Awaitable[Any]],
            max_retries: int = 3,
            backoff_factor: float = 1.5,
            circuit_breaker: Any = None,
            *args,
            **kwargs
    ) -> Any:
        """
        지수 백오프 및 회로 차단기를 사용한 재시도 로직으로 함수를 실행합니다.

        Args:
            func: 실행할 비동기 함수
            max_retries: 최대 재시도 횟수
            backoff_factor: 재시도 간 대기 시간 증가 계수
            circuit_breaker: 선택적 회로 차단기 인스턴스
            *args, **kwargs: 함수에 전달할 인수

        Returns:
            Any: 함수 실행 결과

        Raises:
            Exception: 재시도 후에도 함수 실행이 실패한 경우
        """
        retry_count = 0
        last_exception = None

        while retry_count < max_retries:
            # 회로 차단기 확인
            if circuit_breaker and circuit_breaker.is_open():
                await self.log("warning", f"[{self.request_id}] 회로 열림, 함수 호출 건너뜀: {func.__name__}")
                raise RuntimeError(f"서비스 사용 불가: {func.__name__}에 대한 회로가 열려 있음")

            try:
                start_time = time.time()
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time

                # 로그 실행 시간 모니터링
                await self.log("debug", f"[{self.request_id}] 함수 {func.__name__} 완료: {execution_time:.4f}초")

                # 회로 차단기에 성공 기록
                if circuit_breaker:
                    circuit_breaker.record_success()

                return result
            except Exception as e:
                retry_count += 1
                wait_time = backoff_factor ** retry_count
                last_exception = e

                # 회로 차단기에 실패 기록
                if circuit_breaker:
                    circuit_breaker.record_failure(e)

                await self.log(
                    "warning",
                    f"[{self.request_id}] {func.__name__}에 대한 재시도 {retry_count}/{max_retries} "
                    f"{wait_time:.2f}초 후 - 원인: {type(e).__name__}: {str(e)}"
                )

                # 마지막 시도가 아니면 잠시 대기 후 재시도
                if retry_count < max_retries:
                    await asyncio.sleep(wait_time)

        # 모든 재시도 실패
        await self.log("error", f"[{self.request_id}] {func.__name__}에 대한 모든 {max_retries}회 재시도 실패")
        raise last_exception or RuntimeError(f"{func.__name__}에 대한 모든 재시도 실패")
