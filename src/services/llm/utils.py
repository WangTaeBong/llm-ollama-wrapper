"""
LLM 서비스 유틸리티 모듈

LLM 서비스 구성 요소에서 공통적으로 사용되는 유틸리티 함수와 도우미 클래스를 제공합니다.
이 모듈은 시간 측정, 비동기 처리, 재시도 메커니즘, 데이터 변환 등의 기능을 포함합니다.

주요 기능:
- 비동기 함수 재시도 메커니즘
- 시간 제한이 있는 비동기 실행
- 성능 측정 및 벤치마킹
- 데이터 형식 변환 및 검증
- 백그라운드 작업 관리
"""

import asyncio
import functools
import logging
import time
import traceback
from typing import Any, Callable, Coroutine, Dict, List, Optional, TypeVar, Union, Set
from datetime import datetime

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


def async_retry(max_retries: int = 3, backoff_factor: float = 1.5, circuit_breaker: Optional[Any] = None):
    """
    비동기 함수에 대한 재시도 데코레이터

    지수 백오프와 회로 차단기를 통합하여 안정성을 높입니다.

    Args:
        max_retries: 최대 재시도 횟수
        backoff_factor: 재시도 간 대기 시간 증가 계수
        circuit_breaker: 회로 차단기 인스턴스(선택 사항)

    Returns:
        Callable: 재시도 로직이 포함된 데코레이터 함수
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            retry_count = 0
            last_exception = None

            while retry_count < max_retries:
                # 회로 차단기 확인
                if circuit_breaker and circuit_breaker.is_open():
                    logger.warning(f"회로 개방, {func.__name__} 호출 건너뛰기")
                    raise RuntimeError(f"서비스 사용 불가: {func.__name__}에 대한 회로가 열려 있습니다")

                try:
                    start_time = time.time()
                    result = await func(*args, **kwargs)
                    execution_time = time.time() - start_time

                    # 실행 시간 로깅
                    logger.debug(f"함수 {func.__name__} 완료: {execution_time:.4f}초")

                    # 회로 차단기에 성공 기록
                    if circuit_breaker:
                        circuit_breaker.record_success()

                    return result
                except (asyncio.TimeoutError, ConnectionError) as e:
                    # 특정 오류에 대해 회로 차단기에 실패 기록
                    if circuit_breaker:
                        circuit_breaker.record_failure()

                    retry_count += 1
                    wait_time = backoff_factor ** retry_count
                    last_exception = e

                    logger.warning(
                        f"{func.__name__}에 대한 재시도 {retry_count}/{max_retries} "
                        f"{wait_time:.2f}초 후 - 원인: {type(e).__name__}: {str(e)}"
                    )

                    # 재시도 전 대기
                    await asyncio.sleep(wait_time)
                except Exception as e:
                    # 다른 예외에 대해서는 실패 기록만 하고 재시도하지 않음
                    if circuit_breaker:
                        circuit_breaker.record_failure()
                    raise

            # 모든 재시도 실패
            logger.error(f"{func.__name__}에 대한 모든 {max_retries}회 재시도 실패")
            raise last_exception or RuntimeError(f"{func.__name__}에 대한 모든 재시도 실패")

        return wrapper

    return decorator


def fire_and_forget(coro: Coroutine[Any, Any, Any], session_id: str = "unknown", background_tasks: Optional[list] = None) -> asyncio.Task:
    """
    코루틴을 백그라운드에서 비동기적으로 실행합니다.
    결과를 기다리지 않고 주 실행 흐름에 영향을 주지 않습니다.

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
            logger.debug(f"백그라운드 태스크 스택 트레이스: {traceback.format_exc()}")

    task = asyncio.create_task(wrapper())

    # 태스크 추적을 위해 리스트에 추가
    if background_tasks is not None:
        background_tasks.append(task)

        # 완료 시 리스트에서 제거하는 콜백 추가
        task.add_done_callback(
            lambda t: background_tasks.remove(t) if t in background_tasks else None
        )

    return task


def get_performance_timer():
    """
    성능 측정 타이머를 생성합니다.

    Returns:
        Dict: 타이머 기능이 포함된 딕셔너리
    """
    start_time = time.time()
    stages = {}

    def record_stage(stage_name: str) -> float:
        """
        단계 시간을 기록하고 경과 시간을 반환합니다.

        Args:
            stage_name: 성능 단계의 이름

        Returns:
            float: 마지막 단계부터의 경과 시간
        """
        nonlocal start_time
        current_time = time.time()
        elapsed = current_time - start_time
        stages[stage_name] = elapsed
        start_time = current_time
        return elapsed

    def get_total_time() -> float:
        """
        모든 기록된 단계의 총 시간을 반환합니다.

        Returns:
            float: 총 경과 시간
        """
        return sum(stages.values())

    def get_stages() -> Dict[str, float]:
        """
        기록된 모든 단계를 반환합니다.

        Returns:
            Dict[str, float]: 단계 이름과 경과 시간을 포함하는 딕셔너리
        """
        return stages.copy()

    def reset():
        """
        타이머를 초기화합니다.
        """
        nonlocal start_time, stages
        start_time = time.time()
        stages = {}

    return {
        "record_stage": record_stage,
        "get_total_time": get_total_time,
        "get_stages": get_stages,
        "reset": reset
    }


class MemoryCache:
    """
    간단한 메모리 내 캐시 구현

    TTL(Time-To-Live)과 LRU(Least Recently Used) 캐싱 알고리즘을 결합하여
    메모리 효율성과 성능을 최적화합니다.
    """

    def __init__(self, ttl: int = 3600, max_size: int = 1000, cleanup_interval: int = 300):
        """
        메모리 캐시 초기화

        Args:
            ttl: 캐시 항목의 유효 시간(초), 기본값 1시간
            max_size: 최대 캐시 항목 수
            cleanup_interval: 자동 정리 간격(초)
        """
        self._cache = {}  # 캐시 데이터
        self._timestamps = {}  # 항목 타임스탬프
        self._access_times = {}  # 마지막 접근 시간
        self._ttl = ttl
        self._max_size = max_size
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = time.time()
        self._lock = asyncio.Lock()  # 비동기 안전을 위한 락
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size": 0
        }

    async def get(self, key: str) -> Optional[Any]:
        """
        캐시에서 값을 가져옵니다.

        Args:
            key: 캐시 키

        Returns:
            Optional[Any]: 캐시된 값 또는 None (캐시 미스)
        """
        await self._check_cleanup()

        async with self._lock:
            current_time = time.time()

            if key in self._cache:
                # 만료 확인
                if current_time - self._timestamps[key] < self._ttl:
                    # 접근 시간 업데이트
                    self._access_times[key] = current_time
                    self._stats["hits"] += 1
                    return self._cache[key]
                else:
                    # 만료된 항목 제거
                    self._remove(key)
                    self._stats["evictions"] += 1

            self._stats["misses"] += 1
            return None

    async def set(self, key: str, value: Any) -> None:
        """
        값을 캐시에 저장합니다.

        Args:
            key: 캐시 키
            value: 저장할 값
        """
        await self._check_cleanup()

        async with self._lock:
            current_time = time.time()

            # 최대 크기 확인 및 LRU 정책 적용
            if len(self._cache) >= self._max_size and key not in self._cache:
                await self._evict_lru_item()

            # 캐시에 저장
            self._cache[key] = value
            self._timestamps[key] = current_time
            self._access_times[key] = current_time
            self._stats["size"] = len(self._cache)

    async def delete(self, key: str) -> bool:
        """
        캐시에서 항목을 삭제합니다.

        Args:
            key: 캐시 키

        Returns:
            bool: 삭제 성공 여부
        """
        async with self._lock:
            if key in self._cache:
                self._remove(key)
                return True
            return False

    async def clear(self) -> None:
        """
        모든 캐시 항목을 삭제합니다.
        """
        async with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._access_times.clear()
            self._stats["size"] = 0
            self._stats["evictions"] += 1
            logger.debug("캐시가 완전히 비워졌습니다.")
            return False

    async def _check_cleanup(self) -> None:
        """
        필요 시 캐시 자동 정리를 수행합니다.
        """
        current_time = time.time()

        # 정리 간격 확인
        if current_time - self._last_cleanup < self._cleanup_interval:
            return

        async with self._lock:
            try:
                # 기한 만료된 항목 찾기
                expired_keys = [
                    key for key, timestamp in self._timestamps.items()
                    if current_time - timestamp > self._ttl
                ]

                # 만료된 항목 제거
                for key in expired_keys:
                    self._remove(key)
                    self._stats["evictions"] += 1

                self._last_cleanup = current_time
                if expired_keys:
                    logger.debug(f"캐시 자동 정리 완료: {len(expired_keys)}개 항목 제거됨")

            except Exception as e:
                logger.error(f"캐시 정리 중 오류: {str(e)}")

    def _remove(self, key: str) -> None:
        """
        캐시에서 특정 키를 제거합니다.

        Args:
            key: 제거할 캐시 키
        """
        if key in self._cache:
            del self._cache[key]
        if key in self._timestamps:
            del self._timestamps[key]
        if key in self._access_times:
            del self._access_times[key]
        self._stats["size"] = len(self._cache)

    async def _evict_lru_item(self) -> None:
        """
        가장 오래전에 사용된 항목을 제거합니다(LRU 정책).
        """
        if not self._access_times:
            return

        # 가장 오래전에 접근한 키 찾기
        lru_key = min(self._access_times.items(), key=lambda x: x[1])[0]
        self._remove(lru_key)
        self._stats["evictions"] += 1
        logger.debug(f"LRU 정책으로 캐시 항목 제거: {lru_key}")

    def get_stats(self) -> Dict[str, int]:
        """
        캐시 통계를 반환합니다.

        Returns:
            Dict[str, int]: 캐시 통계 정보
        """
        return self._stats.copy()


def format_time(timestamp: Optional[float] = None) -> str:
    """
    타임스탬프를 사람이 읽기 쉬운 형식으로 변환합니다.

    Args:
        timestamp: 변환할 타임스탬프, None이면 현재 시간 사용

    Returns:
        str: 포맷팅된 시간 문자열
    """
    if timestamp is None:
        timestamp = time.time()

    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def get_today_formatted(format_str: str = "%Y년 %m월 %d일 %A %H시 %M분") -> str:
    """
    현재 날짜와 시간을 지정된 형식으로 반환합니다.

    Args:
        format_str: 날짜 형식 문자열

    Returns:
        str: 포맷팅된 현재 날짜/시간
    """
    import locale
    from datetime import datetime

    # 한국어 로케일 설정 시도
    try:
        locale.setlocale(locale.LC_TIME, 'ko_KR.UTF-8')
    except locale.Error:
        try:
            locale.setlocale(locale.LC_TIME, 'ko_KR')
        except locale.Error:
            # 한국어 로케일을 설정할 수 없는 경우 요일 이름 매핑 사용
            now = datetime.now()
            weekday_names = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
            weekday = weekday_names[now.weekday()]

            # %A를 직접 대체
            format_str = format_str.replace('%A', weekday)
            return now.strftime(format_str)

    # 로케일 설정 성공한 경우
    return datetime.now().strftime(format_str)


def sanitize_input(text: str) -> str:
    """
    입력 텍스트에서 잠재적으로 위험한 문자를 제거합니다.

    Args:
        text: 정제할 입력 텍스트

    Returns:
        str: 정제된 텍스트
    """
    import re

    # XSS 공격 패턴 제거
    text = re.sub(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', '', text, flags=re.IGNORECASE)

    # SQL 인젝션 키워드 제거
    sql_keywords = [
        'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
        'UNION', 'JOIN', 'WHERE', 'HAVING', 'GROUP BY', 'ORDER BY'
    ]

    pattern = r'\b(' + '|'.join(sql_keywords) + r')\b'
    text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # 탈출 문자 및 컨트롤 문자 제거
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)

    return text.strip()


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    텍스트를 지정된 최대 길이로 자릅니다.

    Args:
        text: 자를 텍스트
        max_length: 최대 길이
        suffix: 잘린 텍스트 끝에 추가할 접미사

    Returns:
        str: 잘린 텍스트
    """
    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix


class ContextLogger:
    """
    컨텍스트 정보가 포함된 로깅을 제공하는 클래스

    세션 ID와 같은 컨텍스트 정보를 모든 로그 메시지에 자동으로 포함합니다.
    """

    def __init__(self, session_id: str, logger_name: Optional[str] = None):
        """
        컨텍스트 로거 초기화

        Args:
            session_id: 로그에 포함할 세션 ID
            logger_name: 사용할 로거 이름, None이면 모듈 로거 사용
        """
        self.session_id = session_id
        self.logger = logging.getLogger(logger_name) if logger_name else logger

    def debug(self, message: str, **kwargs):
        """
        DEBUG 수준 로그를 기록합니다.

        Args:
            message: 로그 메시지
            **kwargs: 추가 컨텍스트 데이터
        """
        self._log('debug', message, **kwargs)

    def info(self, message: str, **kwargs):
        """
        INFO 수준 로그를 기록합니다.

        Args:
            message: 로그 메시지
            **kwargs: 추가 컨텍스트 데이터
        """
        self._log('info', message, **kwargs)

    def warning(self, message: str, **kwargs):
        """
        WARNING 수준 로그를 기록합니다.

        Args:
            message: 로그 메시지
            **kwargs: 추가 컨텍스트 데이터
        """
        self._log('warning', message, **kwargs)

    def error(self, message: str, **kwargs):
        """
        ERROR 수준 로그를 기록합니다.

        Args:
            message: 로그 메시지
            **kwargs: 추가 컨텍스트 데이터
        """
        self._log('error', message, **kwargs)

    def _log(self, level: str, message: str, **kwargs):
        """
        지정된 수준에서 컨텍스트 정보가 포함된 로그를 기록합니다.

        Args:
            level: 로그 수준
            message: 로그 메시지
            **kwargs: 추가 컨텍스트 데이터
        """
        # 세션 ID 추가
        formatted_message = f"[{self.session_id}] {message}"

        # 로깅 수준에 따라 로거 메서드 선택
        log_method = getattr(self.logger, level)

        # exc_info 매개변수 별도 처리
        exc_info = kwargs.pop('exc_info', False)

        # 컨텍스트 데이터에 세션 ID 추가
        kwargs['session_id'] = self.session_id

        # 로그 기록
        log_method(formatted_message, exc_info=exc_info, extra=kwargs)
