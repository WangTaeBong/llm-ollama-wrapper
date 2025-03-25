"""
회로 차단기(Circuit Breaker) 모듈

외부 서비스 호출의 안정성과 복원력을 향상시키기 위한 회로 차단기 패턴을 구현합니다.
이 패턴은 연속적인 실패가 발생할 때 추가 요청을 일시적으로 차단하여 시스템 안정성을 보호하고
과부하 상태인 외부 서비스가 복구될 시간을 제공합니다.

주요 기능:
- 연속 실패 임계값 기반의 회로 개방
- 자동 복구를 위한 반-개방(Half-Open) 상태
- 시간 기반 회로 초기화
- 동시성 안전한 상태 관리
- 상태 전환 이벤트 로깅

회로 상태:
- CLOSED: 정상 작동 - 요청이 직접 통과
- OPEN: 회로 개방 - 모든 요청이 즉시 거부됨
- HALF-OPEN: 테스트 상태 - 제한된 요청만 허용하여 복구 여부 확인
"""

import logging
import time
import asyncio
import threading
from typing import Dict, Any, Callable, Optional, List, Union
from enum import Enum, auto
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps

# 로거 설정
logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """회로 차단기의 상태를 나타내는 열거형"""
    CLOSED = auto()  # 정상 작동 상태
    OPEN = auto()  # 회로 차단 상태
    HALF_OPEN = auto()  # 부분 복구 테스트 상태


@dataclass
class CircuitStats:
    """회로 차단기 통계 데이터"""
    total_requests: int = 0  # 총 요청 수
    successful_requests: int = 0  # 성공한 요청 수
    failed_requests: int = 0  # 실패한 요청 수
    rejected_requests: int = 0  # 거부된 요청 수
    state_changes: List[Dict] = None  # 상태 변경 기록
    last_failure_time: float = 0  # 마지막 실패 시간
    last_success_time: float = 0  # 마지막 성공 시간

    def __post_init__(self):
        if self.state_changes is None:
            self.state_changes = []


class CircuitBreaker:
    """
    회로 차단기 클래스

    외부 서비스 호출 시 발생하는 연속적인 실패를 감지하고
    일시적으로 요청을 차단하여 시스템 안정성을 보호합니다.

    Attributes:
        name: 회로 차단기 이름(식별자)
        failure_threshold: 회로를 열기 위한 연속 실패 임계값
        recovery_timeout: 회로를 반-개방 상태로 전환하기 전 대기 시간(초)
        reset_timeout: 회로를 완전히 초기화하기 위한 시간(초)
        half_open_max_calls: 반-개방 상태에서 허용할 최대 호출 수
        exclude_exceptions: 실패 카운트에서 제외할 예외 유형 목록
    """

    # 전역 회로 차단기 레지스트리 - 이름별로 인스턴스 추적
    _registry: Dict[str, 'CircuitBreaker'] = {}

    # 레지스트리 액세스를 위한 쓰레드 안전 락
    _registry_lock = threading.RLock()

    def __init__(
            self,
            name: str,
            failure_threshold: int = 3,
            recovery_timeout: int = 60,
            reset_timeout: int = 300,
            half_open_max_calls: int = 1,
            exclude_exceptions: List[type] = None
    ):
        """
        회로 차단기 초기화

        Args:
            name: 회로 차단기 이름/식별자
            failure_threshold: 회로를 열기 위한 연속 실패 임계값
            recovery_timeout: 회로를 반-개방 상태로 전환하기 전 대기 시간(초)
            reset_timeout: 회로를 완전히 초기화하기 위한 시간(초)
            half_open_max_calls: 반-개방 상태에서 허용할 최대 호출 수
            exclude_exceptions: 실패 카운트에서 제외할 예외 유형 목록
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.reset_timeout = reset_timeout
        self.half_open_max_calls = half_open_max_calls
        self.exclude_exceptions = exclude_exceptions or []

        # 상태 관련 변수
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.open_time = None
        self.last_state_change_time = time.time()
        self.half_open_calls = 0

        # 통계 관련 변수
        self.stats = CircuitStats()

        # 쓰레드 안전성을 위한 락
        self.lock = threading.RLock()

        # 이름 기반 레지스트리에 등록
        with self._registry_lock:
            self._registry[name] = self

        logger.info(f"회로 차단기 '{name}' 초기화 완료 (임계값: {failure_threshold}, 복구 시간: {recovery_timeout}초)")

    def __enter__(self):
        """
        컨텍스트 관리자 진입 메서드

        회로 차단기를 컨텍스트 관리자로 사용할 수 있게 합니다.

        Returns:
            CircuitBreaker: 현재 회로 차단기 인스턴스

        Raises:
            CircuitBreakerOpenError: 회로가 열려 있을 때 발생
        """
        if self.is_open():
            self.stats.rejected_requests += 1
            raise CircuitBreakerOpenError(f"회로 차단기 '{self.name}'가 열려 있습니다.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        컨텍스트 관리자 종료 메서드

        컨텍스트 블록 실행 결과에 따라 성공/실패를 기록합니다.

        Args:
            exc_type: 예외 유형 또는 None
            exc_val: 예외 값 또는 None
            exc_tb: 예외 트레이스백 또는 None

        Returns:
            bool: 예외를 처리했으면 True, 아니면 False
        """
        if exc_type is None:
            self.record_success()
            return False

        # 제외 예외 확인
        if any(isinstance(exc_val, exc) for exc in self.exclude_exceptions):
            return False

        self.record_failure()
        return False  # 예외를 다시 발생시킴

    async def __aenter__(self):
        """
        비동기 컨텍스트 관리자 진입 메서드

        회로 차단기를 비동기 컨텍스트 관리자로 사용할 수 있게 합니다.

        Returns:
            CircuitBreaker: 현재 회로 차단기 인스턴스

        Raises:
            CircuitBreakerOpenError: 회로가 열려 있을 때 발생
        """
        if self.is_open():
            self.stats.rejected_requests += 1
            raise CircuitBreakerOpenError(f"회로 차단기 '{self.name}'가 열려 있습니다.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        비동기 컨텍스트 관리자 종료 메서드

        컨텍스트 블록 실행 결과에 따라 성공/실패를 기록합니다.

        Args:
            exc_type: 예외 유형 또는 None
            exc_val: 예외 값 또는 None
            exc_tb: 예외 트레이스백 또는 None

        Returns:
            bool: 예외를 처리했으면 True, 아니면 False
        """
        if exc_type is None:
            self.record_success()
            return False

        # 제외 예외 확인
        if any(isinstance(exc_val, exc) for exc in self.exclude_exceptions):
            return False

        self.record_failure()
        return False  # 예외를 다시 발생시킴

    def is_open(self) -> bool:
        """
        회로가 열려 있는지 확인하고 자동 복구 상태를 처리합니다.

        Returns:
            bool: 요청을 차단해야 하면 True, 허용해야 하면 False
        """
        with self.lock:
            if self.state == CircuitState.CLOSED:
                return False

            if self.state == CircuitState.OPEN:
                # 복구 타임아웃 확인
                if time.time() - self.open_time > self.recovery_timeout:
                    logger.info(f"회로 차단기 '{self.name}' 상태 전환: OPEN → HALF-OPEN")
                    self._change_state(CircuitState.HALF_OPEN)
                    self.half_open_calls = 0
                    return False
                return True

            # HALF-OPEN 상태
            if self.half_open_calls < self.half_open_max_calls:
                self.half_open_calls += 1
                return False
            return True

    def record_success(self):
        """
        성공적인 호출을 기록합니다.

        HALF-OPEN 상태에서는 회로를 닫고,
        CLOSED 상태에서는 실패 카운터를 재설정합니다.
        """
        with self.lock:
            self.stats.total_requests += 1
            self.stats.successful_requests += 1
            self.stats.last_success_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                logger.info(f"회로 차단기 '{self.name}' 상태 전환: HALF-OPEN → CLOSED (서비스 복구 확인)")
                self._change_state(CircuitState.CLOSED)
                self.failure_count = 0
                self.half_open_calls = 0
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0

    def record_failure(self):
        """
        실패한 호출을 기록합니다.

        HALF-OPEN 상태에서는 회로를 다시 열고,
        CLOSED 상태에서는 실패 카운터를 증가시켜 임계값 도달 시 회로를 엽니다.
        """
        with self.lock:
            self.stats.total_requests += 1
            self.stats.failed_requests += 1
            self.stats.last_failure_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                logger.warning(f"회로 차단기 '{self.name}' 상태 유지: HALF-OPEN → OPEN (복구 실패)")
                self._change_state(CircuitState.OPEN)
                self.open_time = time.time()
            elif self.state == CircuitState.CLOSED:
                self.failure_count += 1
                if self.failure_count >= self.failure_threshold:
                    logger.warning(f"회로 차단기 '{self.name}' 상태 전환: CLOSED → OPEN (임계값 {self.failure_threshold}회 도달)")
                    self._change_state(CircuitState.OPEN)
                    self.open_time = time.time()

    def _change_state(self, new_state: CircuitState):
        """
        회로 상태를 변경하고 상태 변경 이벤트를 기록합니다.

        Args:
            new_state: 새로운 회로 상태
        """
        old_state = self.state
        self.state = new_state
        self.last_state_change_time = time.time()

        # 상태 변경 이벤트 기록
        state_change = {
            'timestamp': time.time(),
            'old_state': old_state.name,
            'new_state': new_state.name,
            'failure_count': self.failure_count
        }
        self.stats.state_changes.append(state_change)

    def reset(self):
        """
        회로 차단기 상태를 초기화합니다.

        강제로 회로를 닫고 모든 카운터를 재설정합니다.
        """
        with self.lock:
            if self.state != CircuitState.CLOSED:
                logger.info(f"회로 차단기 '{self.name}' 수동 초기화: {self.state.name} → CLOSED")
                self._change_state(CircuitState.CLOSED)

            self.failure_count = 0
            self.half_open_calls = 0
            self.open_time = None

    def get_stats(self) -> Dict[str, Any]:
        """
        회로 차단기의 현재 상태와 통계 정보를 반환합니다.

        Returns:
            Dict[str, Any]: 상태 및 통계 정보를 포함하는 딕셔너리
        """
        with self.lock:
            uptime = time.time() - self.last_state_change_time

            return {
                'name': self.name,
                'state': self.state.name,
                'uptime_seconds': uptime,
                'failure_count': self.failure_count,
                'failure_threshold': self.failure_threshold,
                'total_requests': self.stats.total_requests,
                'successful_requests': self.stats.successful_requests,
                'failed_requests': self.stats.failed_requests,
                'rejected_requests': self.stats.rejected_requests,
                'success_rate': (
                    (self.stats.successful_requests / self.stats.total_requests * 100)
                    if self.stats.total_requests > 0 else 0
                ),
                'last_failure': self.stats.last_failure_time,
                'last_success': self.stats.last_success_time,
                'state_changes': len(self.stats.state_changes),
                'last_state_change': self.last_state_change_time
            }

    @classmethod
    def get_instance(cls, name: str) -> Optional['CircuitBreaker']:
        """
        지정된 이름의 회로 차단기 인스턴스를 반환합니다.

        Args:
            name: 찾을 회로 차단기의 이름

        Returns:
            Optional[CircuitBreaker]: 회로 차단기 인스턴스 또는 None
        """
        with cls._registry_lock:
            return cls._registry.get(name)

    @classmethod
    def get_all_instances(cls) -> Dict[str, 'CircuitBreaker']:
        """
        모든 회로 차단기 인스턴스를 반환합니다.

        Returns:
            Dict[str, CircuitBreaker]: 이름으로 매핑된 회로 차단기 인스턴스
        """
        with cls._registry_lock:
            return cls._registry.copy()

    @classmethod
    def reset_all(cls):
        """
        모든 등록된 회로 차단기를 초기화합니다.
        """
        with cls._registry_lock:
            for breaker in cls._registry.values():
                breaker.reset()

            logger.info(f"모든 회로 차단기 초기화 완료 ({len(cls._registry)}개)")

    def call(self, func, *args, **kwargs):
        """
        함수를 회로 차단기로 감싸서 호출합니다.

        Args:
            func: 호출할 함수
            *args: 함수에 전달할 위치 인수
            **kwargs: 함수에 전달할 키워드 인수

        Returns:
            Any: 함수 호출 결과

        Raises:
            CircuitBreakerOpenError: 회로가 열려 있을 때 발생
            Exception: 감싸진 함수에서 발생한 예외
        """
        if self.is_open():
            self.stats.rejected_requests += 1
            raise CircuitBreakerOpenError(f"회로 차단기 '{self.name}'가 열려 있습니다.")

        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            # 제외 예외 확인
            if any(isinstance(e, exc) for exc in self.exclude_exceptions):
                raise

            self.record_failure()
            raise

    async def async_call(self, func, *args, **kwargs):
        """
        비동기 함수를 회로 차단기로 감싸서 호출합니다.

        Args:
            func: 호출할 비동기 함수
            *args: 함수에 전달할 위치 인수
            **kwargs: 함수에 전달할 키워드 인수

        Returns:
            Any: 비동기 함수 호출 결과

        Raises:
            CircuitBreakerOpenError: 회로가 열려 있을 때 발생
            Exception: 감싸진 함수에서 발생한 예외
        """
        if self.is_open():
            self.stats.rejected_requests += 1
            raise CircuitBreakerOpenError(f"회로 차단기 '{self.name}'가 열려 있습니다.")

        try:
            result = await func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            # 제외 예외 확인
            if any(isinstance(e, exc) for exc in self.exclude_exceptions):
                raise

            self.record_failure()
            raise

    def decorator(self, func):
        """
        함수를 회로 차단기로 감싸는 데코레이터를 반환합니다.

        Args:
            func: 감쌀 함수

        Returns:
            Callable: 감싸진 함수
        """
        # 비동기 함수인지 확인
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self.async_call(func, *args, **kwargs)

            return async_wrapper
        else:
            @wraps(func)
            def wrapper(*args, **kwargs):
                return self.call(func, *args, **kwargs)

            return wrapper


class CircuitBreakerOpenError(Exception):
    """회로 차단기가 열려 있을 때 발생하는 예외"""
    pass


# 데코레이터 함수를 사용하여 회로 차단기로 함수 감싸기
def circuit_breaker(
        name: str = None,
        failure_threshold: int = 3,
        recovery_timeout: int = 60,
        reset_timeout: int = 300,
        exclude_exceptions: List[type] = None
):
    """
    함수에 회로 차단기 패턴을 적용하는 데코레이터

    Args:
        name: 회로 차단기 이름 (None이면 함수 이름 사용)
        failure_threshold: 회로를 열기 위한 연속 실패 임계값
        recovery_timeout: 회로를 반-개방 상태로 전환하기 전 대기 시간(초)
        reset_timeout: 회로를 완전히 초기화하기 위한 시간(초)
        exclude_exceptions: 실패 카운트에서 제외할 예외 유형 목록

    Returns:
        Callable: 데코레이터 함수
    """

    def decorator(func):
        breaker_name = name or f"cb_{func.__module__}_{func.__name__}"

        # 기존 회로 차단기 확인 또는 새로 생성
        breaker = CircuitBreaker.get_instance(breaker_name)
        if not breaker:
            breaker = CircuitBreaker(
                name=breaker_name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                reset_timeout=reset_timeout,
                exclude_exceptions=exclude_exceptions
            )

        return breaker.decorator(func)

    return decorator
