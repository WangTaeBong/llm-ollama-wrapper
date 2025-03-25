"""
회로 차단기 모듈

외부 서비스 호출에 대한 회로 차단기 패턴을 구현하여 시스템 안정성을 보장합니다.
연속된 오류가 발생할 경우 회로를 열어 추가 요청을 차단하고, 일정 시간 후 복구를 시도합니다.
"""

import logging
import time
from threading import Lock
from typing import Callable, Dict, List, Optional, Set, Type

from src.common.config_loader import ConfigLoader

# 로거 설정
logger = logging.getLogger(__name__)

# 설정 로드
config = ConfigLoader()
settings = config.get_settings()


class CircuitState:
    """회로 상태 상수"""
    CLOSED = "CLOSED"  # 정상 작동 상태
    OPEN = "OPEN"  # 회로 차단 상태
    HALF_OPEN = "HALF_OPEN"  # 복구 테스트 상태


class CircuitBreakerMetrics:
    """회로 차단기 메트릭 수집 클래스"""

    def __init__(self):
        """메트릭 초기화"""
        self.successful_calls = 0
        self.failed_calls = 0
        self.rejected_calls = 0
        self.state_changes = 0
        self.last_failure_time = None
        self.last_success_time = None
        self.current_state_start_time = time.time()
        self.state_durations = {
            CircuitState.CLOSED: 0.0,
            CircuitState.OPEN: 0.0,
            CircuitState.HALF_OPEN: 0.0
        }
        self.state_history: List[Dict[str, any]] = []

    def record_state_change(self, old_state: str, new_state: str) -> None:
        """상태 변경 기록"""
        now = time.time()
        duration = now - self.current_state_start_time
        self.state_durations[old_state] += duration
        self.current_state_start_time = now
        self.state_changes += 1

        # 이력이 너무 길어지지 않도록 최근 50개 항목만 유지
        if len(self.state_history) >= 50:
            self.state_history.pop(0)

        self.state_history.append({
            'from': old_state,
            'to': new_state,
            'time': now,
            'duration': duration
        })

    def get_metrics(self) -> Dict[str, any]:
        """현재 메트릭 정보 반환"""
        total_calls = self.successful_calls + self.failed_calls + self.rejected_calls
        failure_rate = self.failed_calls / max(total_calls, 1) * 100

        return {
            'total_calls': total_calls,
            'successful_calls': self.successful_calls,
            'failed_calls': self.failed_calls,
            'rejected_calls': self.rejected_calls,
            'failure_rate': failure_rate,
            'state_changes': self.state_changes,
            'state_durations': self.state_durations.copy(),
            'last_failure_time': self.last_failure_time,
            'last_success_time': self.last_success_time
        }


class CircuitBreaker:
    """
    회로 차단기 패턴 구현 클래스

    연속된 실패가 발생할 경우 회로를 열어 추가 요청을 차단하고,
    자동으로 복구를 시도하여 시스템 안정성을 보장합니다.

    회로는 세 가지 상태를 가집니다:
    - CLOSED: 정상 작동 상태, 요청이 통과됨
    - OPEN: 회로가 열린 상태, 모든 요청이 차단됨
    - HALF-OPEN: 테스트 상태, 제한된 요청이 허용되어 서비스 복구 여부를 확인
    """

    def __init__(
            self,
            name: str = "default",
            failure_threshold: Optional[int] = None,
            recovery_timeout: Optional[int] = None,
            reset_timeout: Optional[int] = None,
            excluded_exceptions: Optional[Set[Type[Exception]]] = None,
            state_change_callback: Optional[Callable[[str, str], None]] = None
    ):
        """
        회로 차단기 초기화

        Args:
            name: 회로 차단기 식별자
            failure_threshold: 회로를 열기 전 연속 실패 허용 횟수 (기본값: 설정 파일 또는 3)
            recovery_timeout: 복구 시도 전 대기 시간(초) (기본값: 설정 파일 또는 60)
            reset_timeout: 회로를 완전히 재설정하기 전 대기 시간(초) (기본값: 설정 파일 또는 300)
            excluded_exceptions: 실패로 간주하지 않을 예외 타입 집합
            state_change_callback: 상태 변경 시 호출할 콜백 함수
        """
        self.name = name

        # 설정 파일에서 값 로드 또는 기본값 사용
        self.failure_threshold = failure_threshold if failure_threshold is not None else self._get_config_value(
            'failure_threshold', 3)
        self.recovery_timeout = recovery_timeout if recovery_timeout is not None else self._get_config_value(
            'recovery_timeout', 60)
        self.reset_timeout = reset_timeout if reset_timeout is not None else self._get_config_value('reset_timeout',
                                                                                                    300)

        self.failure_count = 0
        self.open_time = None
        self.state = CircuitState.CLOSED
        self.half_open_calls = 0
        self.lock = Lock()
        self.excluded_exceptions = excluded_exceptions or set()
        self.state_change_callback = state_change_callback
        self.metrics = CircuitBreakerMetrics()

        logger.info(f"회로 차단기 '{name}' 초기화: 실패 임계값={self.failure_threshold}, "
                    f"복구 타임아웃={self.recovery_timeout}초, 재설정 타임아웃={self.reset_timeout}초")

    @staticmethod
    def _get_config_value(key: str, default_value: int) -> int:
        """
        설정 파일에서 회로 차단기 설정 값을 가져옵니다.

        Args:
            key: 설정 키
            default_value: 기본값 (설정 파일에 없는 경우 사용)

        Returns:
            int: 설정 값
        """
        try:
            return getattr(settings.circuit_breaker, key, default_value)
        except (AttributeError, ValueError):
            logger.warning(f"회로 차단기 설정 '{key}'를 찾을 수 없습니다. 기본값 {default_value} 사용")
            return default_value

    def is_open(self) -> bool:
        """
        회로가 열려 있는지 확인하고 자동 복구를 처리합니다.

        Returns:
            bool: 요청을 차단해야 하면 True, 허용해야 하면 False
        """
        with self.lock:
            if self.state == CircuitState.CLOSED:
                return False

            if self.state == CircuitState.OPEN:
                # 복구 타임아웃 경과 여부 확인
                if time.time() - self.open_time > self.recovery_timeout:
                    self._change_state(CircuitState.HALF_OPEN)
                    return False

                # 요청 거부 메트릭 증가
                self.metrics.rejected_calls += 1
                return True

            # HALF-OPEN 상태에서는 제한된 호출 허용
            if self.half_open_calls < 1:
                self.half_open_calls += 1
                return False

            # 추가 요청 거부 메트릭 증가
            self.metrics.rejected_calls += 1
            return True

    def record_success(self) -> None:
        """
        성공적인 서비스 호출을 기록합니다.

        HALF-OPEN 상태에서는 회로를 닫습니다.
        CLOSED 상태에서는 실패 카운터를 재설정합니다.
        """
        with self.lock:
            # 메트릭 업데이트
            self.metrics.successful_calls += 1
            self.metrics.last_success_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                self._change_state(CircuitState.CLOSED)
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0

    def record_failure(self, exception: Optional[Exception] = None) -> None:
        """
        실패한 서비스 호출을 기록합니다.

        Args:
            exception: 발생한 예외 (제외 목록 확인용)

        Note:
            제외된 예외 타입은 실패로 간주하지 않습니다.
        """
        # 제외된 예외 타입인 경우 무시
        if exception and type(exception) in self.excluded_exceptions:
            logger.debug(f"회로 차단기 '{self.name}': 제외된 예외 타입 {type(exception).__name__}, 실패로 간주하지 않음")
            return

        with self.lock:
            # 메트릭 업데이트
            self.metrics.failed_calls += 1
            self.metrics.last_failure_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                self._change_state(CircuitState.OPEN)
            elif self.state == CircuitState.CLOSED:
                self.failure_count += 1
                if self.failure_count >= self.failure_threshold:
                    self._change_state(CircuitState.OPEN)

    def get_state(self) -> str:
        """현재 회로 상태 반환"""
        return self.state

    def get_metrics(self) -> Dict[str, any]:
        """현재 메트릭 정보 반환"""
        return self.metrics.get_metrics()

    def reset(self) -> None:
        """회로 차단기 상태를 초기화합니다."""
        with self.lock:
            old_state = self.state
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.open_time = None
            self.half_open_calls = 0

            # 상태 변경 메트릭 업데이트
            self.metrics.record_state_change(old_state, CircuitState.CLOSED)

            # 콜백 호출
            if self.state_change_callback:
                try:
                    self.state_change_callback(old_state, CircuitState.CLOSED)
                except Exception as e:
                    logger.error(f"회로 차단기 '{self.name}': 상태 변경 콜백 호출 중 오류: {e}")

            logger.info(f"회로 차단기 '{self.name}'가 수동으로 초기화되었습니다")

    def _change_state(self, new_state: str) -> None:
        """
        회로 상태 변경 및 관련 작업 수행

        Args:
            new_state: 새 상태
        """
        old_state = self.state
        self.state = new_state

        if new_state == CircuitState.OPEN:
            self.open_time = time.time()
            logger.warning(f"회로 차단기 '{self.name}'가 열림 - 실패 임계값에 도달")
        elif new_state == CircuitState.HALF_OPEN:
            self.half_open_calls = 0
            logger.info(f"회로 차단기 '{self.name}'가 반열림(half-open) 상태로 전환됨")
        elif new_state == CircuitState.CLOSED:
            self.failure_count = 0
            self.open_time = None
            self.half_open_calls = 0
            logger.info(f"회로 차단기 '{self.name}'가 닫힘 - 서비스가 복구됨")

        # 상태 변경 메트릭 업데이트
        self.metrics.record_state_change(old_state, new_state)

        # 콜백 호출
        if self.state_change_callback:
            try:
                self.state_change_callback(old_state, new_state)
            except Exception as e:
                logger.error(f"회로 차단기 '{self.name}': 상태 변경 콜백 호출 중 오류: {e}")

    @classmethod
    def create_from_config(cls, name: str = "default", **kwargs) -> 'CircuitBreaker':
        """
        설정 파일의 값을 사용하여 CircuitBreaker 인스턴스를 생성합니다.

        Args:
            name: 회로 차단기 식별자
            **kwargs: 설정을 재정의할 추가 매개변수

        Returns:
            CircuitBreaker: 생성된 회로 차단기 인스턴스
        """
        return cls(name=name, **kwargs)