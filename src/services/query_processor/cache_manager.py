"""
쿼리 캐시 관리 모듈

쿼리 처리를 위한 효율적인 캐싱 기능을 제공합니다.
"""

import hashlib
import json
import logging
import time
from typing import Any, Optional, Dict

# 로거 설정
logger = logging.getLogger(__name__)


class QueryCache:
    """
    쿼리 캐시 관리 클래스

    싱글톤 패턴을 활용한 전역 캐시 인스턴스를 제공하여 메모리 사용을 최적화하고
    쿼리 처리 결과의 중복 계산을 방지합니다.
    """

    _instance = None

    @classmethod
    def get_instance(cls, ttl: int = 3600):
        """
        싱글톤 인스턴스를 반환합니다.

        Args:
            ttl: 캐시 항목의 유효 시간(초)

        Returns:
            QueryCache: 싱글톤 캐시 인스턴스
        """
        if cls._instance is None:
            cls._instance = cls(ttl)
        return cls._instance

    def __init__(self, ttl: int = 3600):
        """
        쿼리 캐시 관리자 초기화

        Args:
            ttl: 캐시 Time-To-Live(초)
        """
        # 이미 인스턴스가 있는 경우 중복 생성 방지
        if QueryCache._instance is not None:
            raise RuntimeError("QueryCache는 싱글톤 클래스입니다. get_instance() 메서드를 사용하세요.")

        self.cache = {}
        self.timestamps = {}
        self.stats = {
            "hits": 0,
            "misses": 0,
            "size": 0,
            "evictions": 0
        }
        self.ttl = ttl
        self.max_size = 1000  # 최대 캐시 항목 수

        # logger.debug("쿼리 캐시 관리자가 초기화되었습니다")

    def get(self, key: str) -> Optional[Any]:
        """
        캐시에서 값을 가져옵니다.

        Args:
            key: 캐시 키

        Returns:
            Any: 캐시된 값 또는 None (캐시 미스 또는 만료)
        """
        self._clean_expired()

        if key in self.cache:
            current_time = time.time()

            # 캐시 항목이 여전히 유효한지 확인
            if current_time - self.timestamps.get(key, 0) < self.ttl:
                logger.debug(f"캐시 적중: {key}")
                self.stats["hits"] += 1
                self._update_timestamp(key)
                return self.cache[key]
            else:
                # 만료된 캐시 항목 제거
                self._remove(key)
                self.stats["evictions"] += 1

        self.stats["misses"] += 1
        return None

    def set(self, key: str, value: Any) -> None:
        """
        캐시에 항목을 저장합니다.

        Args:
            key: 캐시 키
            value: 저장할 값
        """
        # 캐시 크기 제한 확인
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict_lru_item()

        self.cache[key] = value
        self._update_timestamp(key)
        self.stats["size"] = len(self.cache)

    def _update_timestamp(self, key: str) -> None:
        """
        항목의 타임스탬프를 현재 시간으로 업데이트합니다.

        Args:
            key: 업데이트할 키
        """
        self.timestamps[key] = time.time()

    def _remove(self, key: str) -> None:
        """
        캐시에서 항목을 제거합니다.

        Args:
            key: 제거할 키
        """
        if key in self.cache:
            del self.cache[key]
        if key in self.timestamps:
            del self.timestamps[key]
        self.stats["size"] = len(self.cache)

    def _evict_lru_item(self) -> None:
        """
        가장 오래전에 사용된 항목을 제거합니다(LRU 정책).
        """
        if not self.timestamps:
            return

        # 가장 오래된 타임스탬프를 가진 키 찾기
        oldest_key = min(self.timestamps.items(), key=lambda x: x[1])[0]
        self._remove(oldest_key)
        self.stats["evictions"] += 1
        logger.debug(f"LRU 정책으로 캐시 항목 제거: {oldest_key}")

    def _clean_expired(self) -> None:
        """
        만료된 캐시 항목을 정리합니다.
        주기적으로 호출되어 메모리를 관리합니다.
        """
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.timestamps.items()
            if current_time - timestamp > self.ttl
        ]

        for key in expired_keys:
            self._remove(key)
            self.stats["evictions"] += 1

    def clear(self) -> None:
        """모든 캐시 항목을 지웁니다."""
        self.cache.clear()
        self.timestamps.clear()
        self.stats["size"] = 0
        self.stats["evictions"] += len(self.cache)
        logger.debug("쿼리 캐시를 모두 지웠습니다")

    def get_stats(self) -> Dict[str, int]:
        """
        캐시 통계를 반환합니다.

        Returns:
            Dict[str, int]: 캐시 통계 정보
        """
        return self.stats

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
