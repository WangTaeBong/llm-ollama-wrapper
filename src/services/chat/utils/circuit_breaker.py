"""
서킷 브레이커 패턴 모듈

외부 서비스 호출의 안정성을 향상시키는 서킷 브레이커 패턴을 구현합니다.
"""

import asyncio
import logging
import time
from typing import Callable, Any, Optional, Dict, List
from enum import Enum

# 로거 설정
logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """서킷 브레이커 상태 열거형"""
    CLOSED = "CLOSED"  # 정상 작동 - 모든 요청 전달
    OPEN = "OPEN"  # 오픈 상태 - 모든 요청 차단
    HALF_OPEN = "HALF_OPEN"  # 반개방 상태 - 일부 요청 허용하여 복구 확인


class CircuitBreaker:
    """
    서킷 브레이커 패턴 구현

    연속된 실패가 발생하면 회로를 열어 더 이상의 요청을 차단하고,
    일정 시간이 지난 후 일부 요청을 허용하여 시스템 복구 여부를 확인합니다.
    """

    def __init__(
            self,
            name: str,
            failure_threshold: int = 3,
            recovery_timeout: int = 60,
            reset_timeout: int = 300
    ):
        """
        서킷 브레이커 초기화

        Args:
            name: 서킷 브레이커 식별자
            failure_threshold: 회로를 열기 위한 연속 실패 수
            recovery_timeout: 반개방 상태로 전환하기 전 대기 시간(초)
            reset_timeout: 완전 재설정 시간(초)
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.reset_timeout = reset_timeout

        # 상태 관리
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._open_time = None
        self._half_open_calls = 0
        self._last_success_time = None
        self._last_failure_time = None

        # 비동기 락
        self._lock = asyncio.Lock()

        # 메트릭
        self._metrics = {
            "success_count": 0,
            "failure_count": 0,
            "rejection_count": 0,
            "state_changes": 0,
        }

        logger.debug(f"서킷 브레이커 '{name}' 초기화됨")

    def is_open(self) -> bool:
        """
        현재 회로가 열려있는지 확인하고 상태 전환을 처리합니다.

        Returns:
            bool: 요청이 차단되어야 하면 True, 허용되어야 하면 False
        """
        current_state = self._state
        current_time = time.time()

        # 회로가 닫혀있으면 요청 허용
        if current_state == CircuitState.CLOSED:
            return False

        # 회로가 완전히 열린 상태
        if current_state == CircuitState.OPEN:
            # 복구 타임아웃 확인
            if self._open_time and (current_time - self._open_time) > self.recovery_timeout:
                # 반개방 상태로 전환
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                self._metrics["state_changes"] += 1
                logger.info(f"서킷 브레이커 '{self.name}'가 반개방 상태로 전환됨")
                return False
            return True

        # 반개방 상태 - 제한된 요청만 허용
        if self._half_open_calls < 1:
            # 첫 번째 테스트 요청 허용
            self._half_open_calls += 1
            return False
        return True

    def record_success(self) -> None:
        """
        성공한 요청을 기록합니다.

        반개방 상태에서 성공하면 회로를 닫고,
        닫힌 상태에서는 실패 카운터를 초기화합니다.
        """
        current_state = self._state

        # 메트릭 업데이트
        self._metrics["success_count"] += 1
        self._last_success_time = time.time()

        # 반개방 상태에서 성공 시 회로 닫기
        if current_state == CircuitState.HALF_OPEN:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._open_time = None
            self._half_open_calls = 0
            self._metrics["state_changes"] += 1
            logger.info(f"서킷 브레이커 '{self.name}'가 닫힘 - 서비스가 회복됨")

        # 닫힌 상태에서 실패 카운터 초기화
        elif current_state == CircuitState.CLOSED:
            self._failure_count = 0

    def record_failure(self) -> None:
        """
        실패한 요청을 기록합니다.

        반개방 상태에서 실패하면 다시 회로를 열고,
        닫힌 상태에서는 실패 카운터를 증가시켜 임계값 도달 시 회로를 엽니다.
        """
        current_state = self._state
        current_time = time.time()

        # 메트릭 업데이트
        self._metrics["failure_count"] += 1
        self._last_failure_time = current_time

        # 반개방 상태에서 실패 시 회로 다시 열기
        if current_state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._open_time = current_time
            self._metrics["state_changes"] += 1
            logger.warning(f"서킷 브레이커 '{self.name}'가 다시 열림 - 서비스가 여전히 실패 중")

        # 닫힌 상태에서 실패 처리
        elif current_state == CircuitState.CLOSED:
            self._failure_count += 1
            # 임계값 도달 시 회로 열기
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._open_time = current_time
                self._metrics["state_changes"] += 1
                logger.warning(f"서킷 브레이커 '{self.name}'가 열림 - 연속 {self._failure_count}회 실패")

    async def execute(self, func, *args, **kwargs) -> Any:
        """
        함수 실행을 서킷 브레이커로 보호합니다.

        Args:
            func: 실행할 함수 또는 코루틴
            *args: 함수 인수
            **kwargs: 함수 키워드 인수

        Returns:
            Any: 함수 실행 결과

        Raises:
            RuntimeError: 회로가 열려있을 때
            Exception: 함수 실행 중 발생한 예외
        """
        # 비동기 락으로 상태 확인
        async with self._lock:
            # 회로가 열려있는지 확인
            if self.is_open():
                self._metrics["rejection_count"] += 1
                raise RuntimeError(f"서비스를 사용할 수 없음: 서킷 브레이커 '{self.name}'가 열려 있습니다")

        try:
            # 함수 유형에 따라 실행 방식 결정
            if asyncio.iscoroutinefunction(func):
                # 비동기 함수 실행
                result = await func(*args, **kwargs)
            else:
                # 동기 함수 실행
                result = await asyncio.to_thread(func, *args, **kwargs)

            # 비동기 락으로 성공 기록
            async with self._lock:
                self.record_success()

            return result

        except Exception as e:
            # 비동기 락으로 실패 기록
            async with self._lock:
                self.record_failure()
            raise

    def reset(self) -> None:
        """서킷 브레이커 상태를 초기화합니다."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._open_time = None
        self._half_open_calls = 0
        logger.info(f"서킷 브레이커 '{self.name}' 상태가 초기화됨")

    @property
    def state(self) -> str:
        """현재 회로 상태를 문자열로 반환합니다."""
        return self._state.value

    def get_metrics(self) -> Dict[str, Any]:
        """
        성능 메트릭 조회

        Returns:
            Dict[str, Any]: 성능 메트릭
        """
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self._failure_count,
            "metrics": self._metrics,
            "last_success": self._last_success_time,
            "last_failure": self._last_failure_time,
            "open_time": self._open_time
        }


class CircuitBreakerRegistry:
    """
    서킷 브레이커 레지스트리

    애플리케이션에서 여러 서킷 브레이커를 중앙에서 관리합니다.
    """

    # 싱글톤 인스턴스
    _instance = None

    # 등록된 브레이커 목록
    _breakers: Dict[str, CircuitBreaker] = {}

    @classmethod
    def get_instance(cls):
        """
        싱글톤 인스턴스 반환

        Returns:
            CircuitBreakerRegistry: 레지스트리 인스턴스
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, breaker: CircuitBreaker) -> None:
        """
        서킷 브레이커 등록

        Args:
            breaker: 등록할 서킷 브레이커
        """
        self._breakers[breaker.name] = breaker
        logger.debug(f"서킷 브레이커 '{breaker.name}' 등록됨")

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """
        이름으로 서킷 브레이커 조회

        Args:
            name: 서킷 브레이커 이름

        Returns:
            Optional[CircuitBreaker]: 서킷 브레이커 또는 None
        """
        return self._breakers.get(name)

    def reset_all(self) -> None:
        """모든 서킷 브레이커 초기화"""
        for breaker in self._breakers.values():
            breaker.reset()
        logger.info("모든 서킷 브레이커 초기화됨")

    def get_all_metrics(self) -> List[Dict[str, Any]]:
        """
        모든 서킷 브레이커의 메트릭 조회

        Returns:
            List[Dict[str, Any]]: 서킷 브레이커별 메트릭
        """
        return [breaker.get_metrics() for breaker in self._breakers.values()]

    def get_open_breakers(self) -> List[str]:
        """
        열린 상태인 서킷 브레이커 이름 목록 조회

        Returns:
            List[str]: 열린 상태의 서킷 브레이커 이름
        """
        return [name for name, breaker in self._breakers.items()
                if breaker.state == CircuitState.OPEN.value]
