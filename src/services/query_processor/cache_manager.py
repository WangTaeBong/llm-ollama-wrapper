"""
쿼리 캐시 관리 모듈

쿼리 처리를 위한 캐싱 기능을 제공합니다.
"""

import hashlib
import json
import logging
import time
from typing import Any, Optional, Dict

# 로거 설정
logger = logging.getLogger(__name__)


class QueryCacheManager:
    """
    쿼리 캐시 관리 클래스

    효율적인 쿼리 처리를 위한 캐싱 메커니즘을 제공합니다.
    """

    def __init__(self, ttl: int = 3600):
        """
        쿼리 캐시 관리자 초기화

        Args:
            ttl: 캐시 Time-To-Live(초)
        """
        self.cache = {}
        self.cache_timestamps = {}
        self.ttl = ttl

        logger.debug("쿼리 캐시 관리자가 초기화되었습니다")

    def get(self, key: str) -> Optional[Any]:
        """
        캐시에서 값을 가져옵니다.

        Args:
            key: 캐시 키

        Returns:
            Any: 캐시된 값 또는 None (캐시 미스 또는 만료)
        """
        current_time = time.time()

        if key in self.cache:
            # 캐시 항목이 여전히 유효한지 확인
            if current_time - self.cache_timestamps.get(key, 0) < self.ttl:
                logger.debug(f"캐시 적중: {key}")
                return self.cache[key]
            else:
                # 만료된 캐시 항목 제거
                self._remove(key)

        return None

    def set(self, key: str, value: Any) -> None:
        """
        캐시에 항목을 저장합니다.

        Args:
            key: 캐시 키
            value: 저장할 값
        """
        self.cache[key] = value
        self.cache_timestamps[key] = time.time()

    def _remove(self, key: str) -> None:
        """
        캐시에서 항목을 제거합니다.

        Args:
            key: 제거할 키
        """
        if key in self.cache:
            del self.cache[key]
        if key in self.cache_timestamps:
            del self.cache_timestamps[key]

    def clear(self) -> None:
        """모든 캐시 항목을 지웁니다."""
        self.cache.clear()
        self.cache_timestamps.clear()
        logger.debug("쿼리 캐시를 모두 지웠습니다")

    @staticmethod
    def create_key(data: Any) -> str:
        """
        캐시 키를 생성하는 유틸리티 메서드

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
