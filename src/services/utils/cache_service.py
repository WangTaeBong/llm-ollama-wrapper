"""
캐싱 서비스 모듈

애플리케이션 전체에서 사용되는 다양한 캐싱 메커니즘을 제공합니다.
TTL(Time-To-Live) 기반 캐싱, 키 관리, 그리고 자동 정리 기능이 포함되어 있습니다.
"""

import hashlib
import json
import logging
import time
from typing import Dict, Any, List, Optional, TypeVar, Generic, Callable, Tuple, Union

from cachetools import TTLCache

# 로거 설정
logger = logging.getLogger(__name__)

# 캐시 값 타입 변수
T = TypeVar('T')


class CacheService:
    """
    애플리케이션 전체에서 사용되는 캐싱 서비스

    여러 유형의 데이터를 위한 TTL 기반 캐싱 메커니즘을 제공합니다.
    캐시 키 관리, 자동 정리, 그리고 다양한 캐시 유형에 대한 접근 기능이 포함되어 있습니다.
    """

    # 기본 TTL 값
    DEFAULT_TTL = 3600  # 1시간

    # 캐시 컨테이너
    _caches: Dict[str, TTLCache] = {}

    # 기본 캐시 구성
    _default_configs = {
        "chain": {"maxsize": 50, "ttl": 1800},  # 30분
        "response": {"maxsize": 200, "ttl": 900},  # 15분
        "prompt": {"maxsize": 100, "ttl": 3600},  # 1시간
        "document": {"maxsize": 100, "ttl": 1800},  # 30분
        "general": {"maxsize": 100, "ttl": 1800},  # 30분
    }

    # 마지막 정리 시간
    _last_cleanup_time = time.time()

    # 정리 주기
    _cleanup_interval = 3600  # 1시간마다 정리

    # 초기화 플래그
    _initialized = False

    @classmethod
    def initialize(cls, configs: Optional[Dict[str, Dict[str, int]]] = None) -> None:
        """
        캐시 서비스를 초기화합니다.

        Args:
            configs: 캐시 유형별 구성 (선택 사항)
        """
        if cls._initialized:
            return

        # 구성 병합
        final_configs = cls._default_configs.copy()
        if configs:
            for cache_type, config in configs.items():
                if cache_type in final_configs:
                    final_configs[cache_type].update(config)
                else:
                    final_configs[cache_type] = config

        # 캐시 초기화
        for cache_type, config in final_configs.items():
            cls._caches[cache_type] = TTLCache(
                maxsize=config.get("maxsize", 100),
                ttl=config.get("ttl", cls.DEFAULT_TTL)
            )

        cls._initialized = True
        logger.info(f"캐시 서비스가 초기화되었습니다 - {len(cls._caches)}개 캐시 유형 구성됨")

    @classmethod
    def ensure_initialized(cls) -> None:
        """캐시 서비스가 초기화되었는지 확인하고, 그렇지 않으면 초기화합니다."""
        if not cls._initialized:
            cls.initialize()

    @classmethod
    def get(cls, cache_type: str, key: str, default: Any = None) -> Any:
        """
        지정된 캐시에서 값을 검색합니다.

        Args:
            cache_type: 캐시 유형
            key: 캐시 키
            default: 캐시 미스 시 반환할 기본값

        Returns:
            캐시된 값 또는 기본값
        """
        cls.ensure_initialized()

        # 캐시 유형 확인
        if cache_type not in cls._caches:
            logger.warning(f"알 수 없는 캐시 유형: {cache_type}, 'general' 사용")
            cache_type = "general"

        # 캐시 확인
        cache = cls._caches[cache_type]
        if key in cache:
            logger.debug(f"캐시 적중: {cache_type}:{key}")
            return cache.get(key)

        logger.debug(f"캐시 미스: {cache_type}:{key}")
        return default

    @classmethod
    def set(cls, cache_type: str, key: str, value: Any) -> None:
        """
        지정된 캐시에 값을 저장합니다.

        Args:
            cache_type: 캐시 유형
            key: 캐시 키
            value: 저장할 값
        """
        cls.ensure_initialized()

        # 캐시 유형 확인
        if cache_type not in cls._caches:
            logger.warning(f"알 수 없는 캐시 유형: {cache_type}, 'general' 사용")
            cache_type = "general"

        # 캐시에 저장
        cls._caches[cache_type][key] = value
        logger.debug(f"캐시 설정: {cache_type}:{key}")

        # 자동 정리 확인
        cls._check_auto_cleanup()

    @classmethod
    def delete(cls, cache_type: str, key: str) -> bool:
        """
        지정된 캐시에서 항목을 삭제합니다.

        Args:
            cache_type: 캐시 유형
            key: 삭제할 키

        Returns:
            bool: 삭제 성공 여부
        """
        cls.ensure_initialized()

        # 캐시 유형 확인
        if cache_type not in cls._caches:
            return False

        # 항목 삭제
        cache = cls._caches[cache_type]
        if key in cache:
            del cache[key]
            logger.debug(f"캐시 항목 삭제됨: {cache_type}:{key}")
            return True

        return False

    @classmethod
    def clear(cls, cache_type: Optional[str] = None) -> None:
        """
        지정된 캐시 또는 모든 캐시를 지웁니다.

        Args:
            cache_type: 지울 캐시 유형 (None이면 모든 캐시)
        """
        cls.ensure_initialized()

        if cache_type:
            # 지정된 캐시만 지우기
            if cache_type in cls._caches:
                size = len(cls._caches[cache_type])
                cls._caches[cache_type].clear()
                logger.info(f"캐시 지움: {cache_type} ({size}개 항목)")
        else:
            # 모든 캐시 지우기
            for cache_type, cache in cls._caches.items():
                size = len(cache)
                cache.clear()
                logger.info(f"캐시 지움: {cache_type} ({size}개 항목)")

    @classmethod
    def get_or_create(cls, cache_type: str, key: str,
                      creator_func: Callable[[], T]) -> T:
        """
        캐시에서 값을 가져오거나, 없으면 생성하여 캐싱합니다.

        Args:
            cache_type: 캐시 유형
            key: 캐시 키
            creator_func: 값이 없을 때 호출할 생성 함수

        Returns:
            캐시된 값 또는 새로 생성된 값
        """
        cls.ensure_initialized()

        # 캐시 확인
        value = cls.get(cache_type, key)
        if value is not None:
            return value

        # 값 생성 및 캐싱
        value = creator_func()
        cls.set(cache_type, key, value)
        return value

    @classmethod
    async def get_or_create_async(cls, cache_type: str, key: str,
                                  creator_func: Callable[[], Any]) -> Any:
        """
        캐시에서 값을 가져오거나, 없으면 비동기적으로 생성하여 캐싱합니다.

        Args:
            cache_type: 캐시 유형
            key: 캐시 키
            creator_func: 값이 없을 때 호출할 비동기 생성 함수

        Returns:
            캐시된 값 또는 새로 생성된 값
        """
        cls.ensure_initialized()

        # 캐시 확인
        value = cls.get(cache_type, key)
        if value is not None:
            return value

        # 값 생성 및 캐싱
        value = await creator_func()
        cls.set(cache_type, key, value)
        return value

    @classmethod
    def create_key(cls, data: Any) -> str:
        """
        캐시 키를 생성합니다.

        다양한 데이터 유형에 대해 일관된 해시 키를 생성합니다.

        Args:
            data: 키를 생성할 데이터

        Returns:
            str: 생성된 해시 키
        """
        # 데이터 유형에 따라 해시 생성
        if isinstance(data, str):
            return hashlib.md5(data.encode('utf-8')).hexdigest()
        elif isinstance(data, dict):
            # 일관된 해싱을 위해 정렬된 JSON 문자열로 변환
            return hashlib.md5(json.dumps(data, sort_keys=True).encode('utf-8')).hexdigest()
        else:
            # 다른 데이터 유형을 문자열로 변환
            return hashlib.md5(str(data).encode('utf-8')).hexdigest()

    @classmethod
    def get_stats(cls) -> Dict[str, Dict[str, int]]:
        """
        모든 캐시의 통계 정보를 반환합니다.

        Returns:
            Dict: 캐시 유형별 통계 정보
        """
        cls.ensure_initialized()

        stats = {}
        for cache_type, cache in cls._caches.items():
            stats[cache_type] = {
                "size": len(cache),
                "maxsize": cache.maxsize,
                "ttl": cache.ttl
            }

        return stats

    @classmethod
    def _check_auto_cleanup(cls) -> None:
        """
        자동 캐시 정리가 필요한지 확인하고 필요하면 수행합니다.
        이 메서드는 내부적으로 호출되며 주기적인 캐시 유지 관리를 수행합니다.
        """
        current_time = time.time()

        # 정리 주기 확인
        if current_time - cls._last_cleanup_time < cls._cleanup_interval:
            return

        # 모든 캐시 정리 (TTLCache는 내부적으로 만료된 항목을 처리합니다)
        # 추가 정리 로직을 여기에 구현할 수 있습니다

        # 마지막 정리 시간 업데이트
        cls._last_cleanup_time = current_time

        # 통계 로깅
        stats = cls.get_stats()
        stats_str = ", ".join([f"{d_type}: {data['size']}/{data['maxsize']}"
                               for d_type, data in stats.items()])
        logger.info(f"자동 캐시 정리 완료: {stats_str}")

    @classmethod
    def set_cleanup_interval(cls, interval: int) -> None:
        """
        자동 정리 주기를 설정합니다.

        Args:
            interval: 정리 주기(초)
        """
        cls._cleanup_interval = max(300, interval)  # 최소 5분
        logger.info(f"캐시 정리 주기가 {cls._cleanup_interval}초로 설정되었습니다")
