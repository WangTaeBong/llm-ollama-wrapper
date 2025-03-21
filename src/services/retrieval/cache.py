"""
캐시 관리 모듈
===========

검색 결과 및 관련 데이터를 효율적으로 캐싱하는 기능을 제공합니다.

기능:
- TTL 기반 캐싱
- 메모리 효율적인 데이터 저장
- 다양한 형식의 데이터 캐싱 지원
"""

import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional

# 로거 설정
logger = logging.getLogger(__name__)


class CacheManager:
    """
    캐시 관리자 클래스

    API 응답 및 검색 결과의 캐싱을 담당합니다.
    """

    def __init__(self, ttl: int = 3600):
        """
        캐시 관리자 초기화

        Args:
            ttl: 캐시 Time-To-Live (초 단위)
        """
        self.cache = {}
        self.cache_timestamps = {}
        self.ttl = ttl

    def get(self, key: str) -> Optional[Any]:
        """
        캐시에서 값 조회

        Args:
            key: 캐시 키

        Returns:
            캐시된 값 또는 None (캐시 미스 또는 만료)
        """
        current_time = time.time()

        if key in self.cache:
            # 캐시 항목이 유효한지 확인
            if current_time - self.cache_timestamps.get(key, 0) < self.ttl:
                logger.debug(f"캐시 히트: {key}")
                return self.cache[key]
            else:
                # 만료된 캐시 항목 제거
                self._remove(key)

        return None

    def set(self, key: str, value: Any) -> None:
        """
        캐시에 항목 저장

        Args:
            key: 캐시 키
            value: 저장할 값
        """
        self.cache[key] = value
        self.cache_timestamps[key] = time.time()

    def _remove(self, key: str) -> None:
        """
        캐시에서 항목 제거

        Args:
            key: 제거할 키
        """
        if key in self.cache:
            del self.cache[key]
        if key in self.cache_timestamps:
            del self.cache_timestamps[key]

    def clear(self) -> None:
        """모든 캐시 항목 제거"""
        self.cache.clear()
        self.cache_timestamps.clear()

    def size(self) -> int:
        """
        캐시 크기 반환

        Returns:
            int: 캐시 항목 수
        """
        return len(self.cache)

    @staticmethod
    def create_key(data: Any) -> str:
        """
        캐시 키 생성 유틸리티 메서드

        Args:
            data: 키를 생성할 데이터

        Returns:
            str: 생성된 해시 키
        """
        # 데이터 타입에 따른 해시 생성
        if isinstance(data, str):
            return hashlib.md5(data.encode('utf-8')).hexdigest()
        elif isinstance(data, dict):
            # 정렬된 JSON 문자열로 변환하여 일관된 해싱
            return hashlib.md5(json.dumps(data, sort_keys=True).encode('utf-8')).hexdigest()
        else:
            # 다른 데이터 타입을 문자열로 변환
            return hashlib.md5(str(data).encode('utf-8')).hexdigest()
