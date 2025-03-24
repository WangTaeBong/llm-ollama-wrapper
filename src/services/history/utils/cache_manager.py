"""
캐시 관리 모듈

대화 히스토리 처리를 위한 메모리 캐싱 시스템을 제공합니다.
"""

import hashlib
import json
import logging
import time
from typing import Dict, Any, Optional, Set, DefaultDict
from collections import defaultdict
from threading import Lock

# 로거 설정
logger = logging.getLogger(__name__)


class HistoryCacheManager:
    """
    히스토리 캐시 관리 클래스

    대화 이력 및 관련 데이터에 대한 메모리 내 캐싱을 제공합니다.
    스레드 안전한 캐싱 메커니즘을 사용하며, 백그라운드 정리 기능을 구현하여
    메모리 효율성을 유지합니다.
    """

    # 싱글톤 인스턴스
    _instance = None
    _lock = Lock()

    @classmethod
    def get_instance(cls, ttl: int = 3600, max_size: int = 100):
        """
        싱글톤 인스턴스를 반환합니다.

        Args:
            ttl: 캐시 항목의 유효 시간(초)
            max_size: 각 캐시 유형당 최대 항목 수

        Returns:
            HistoryCacheManager: 싱글톤 캐시 인스턴스
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(ttl, max_size)
            return cls._instance

    def __init__(self, ttl: int = 3600, max_size: int = 100):
        """
        히스토리 캐시 관리자 초기화

        Args:
            ttl: 캐시 항목의 유효 시간(초)
            max_size: 각 캐시 유형당 최대 항목 수
        """
        # 이미 인스턴스가 있는 경우 중복 생성 방지
        if HistoryCacheManager._instance is not None:
            raise RuntimeError("HistoryCacheManager는 싱글톤 클래스입니다. get_instance() 메서드를 사용하세요.")

        # 캐시 설정
        self.ttl = ttl
        self.max_size = max_size

        # 다양한 캐시 컨테이너
        self._session_cache: Dict[str, Any] = {}
        self._chain_cache: Dict[str, Any] = {}
        self._message_cache: Dict[str, Any] = {}
        self._response_cache: Dict[str, Any] = {}

        # 캐시 타임스탬프
        self._timestamps: DefaultDict[str, Dict[str, float]] = defaultdict(dict)

        # 캐시 통계
        self.stats = {
            "hits": defaultdict(int),
            "misses": defaultdict(int),
            "evictions": defaultdict(int),
            "size": defaultdict(int)
        }

        # 캐시 정리 설정
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # 5분마다 자동 정리

        # 메모리 사용량 추적용 집합
        self._processed_message_ids: Set[str] = set()
        self._processed_inputs: Set[str] = set()

    def get(self, cache_type: str, key: str) -> Optional[Any]:
        """
        캐시에서 값을 가져옵니다.

        Args:
            cache_type: 캐시 유형 ("session", "chain", "message", "response")
            key: 캐시 키

        Returns:
            Optional[Any]: 캐시된 값 또는 None (캐시 미스 또는 만료)
        """
        self._check_cleanup()

        cache = self._get_cache_by_type(cache_type)
        if not cache:
            return None

        current_time = time.time()

        if key in cache:
            # 캐시 항목이 여전히 유효한지 확인
            if current_time - self._timestamps[cache_type].get(key, 0) < self.ttl:
                logger.debug(f"캐시 적중: {cache_type}:{key}")
                self.stats["hits"][cache_type] += 1
                return cache[key]
            else:
                # 만료된 캐시 항목 제거
                self._remove(cache_type, key)
                self.stats["evictions"][cache_type] += 1

        self.stats["misses"][cache_type] += 1
        return None

    def set(self, cache_type: str, key: str, value: Any) -> None:
        """
        캐시에 항목을 저장합니다.

        Args:
            cache_type: 캐시 유형 ("session", "chain", "message", "response")
            key: 캐시 키
            value: 저장할 값
        """
        self._check_cleanup()

        cache = self._get_cache_by_type(cache_type)
        if not cache:
            return

        # 캐시 크기 제한 확인
        if len(cache) >= self.max_size and key not in cache:
            self._evict_lru_item(cache_type)

        # 캐시에 저장 및 타임스탬프 업데이트
        cache[key] = value
        self._timestamps[cache_type][key] = time.time()
        self.stats["size"][cache_type] = len(cache)

    def _get_cache_by_type(self, cache_type: str) -> Optional[Dict[str, Any]]:
        """
        캐시 유형에 따른 캐시 컨테이너를 반환합니다.

        Args:
            cache_type: 캐시 유형

        Returns:
            Optional[Dict[str, Any]]: 해당 유형의 캐시 컨테이너 또는 None
        """
        if cache_type == "session":
            return self._session_cache
        elif cache_type == "chain":
            return self._chain_cache
        elif cache_type == "message":
            return self._message_cache
        elif cache_type == "response":
            return self._response_cache
        else:
            logger.warning(f"알 수 없는 캐시 유형: {cache_type}")
            return None

    def _remove(self, cache_type: str, key: str) -> None:
        """
        캐시에서 항목을 제거합니다.

        Args:
            cache_type: 캐시 유형
            key: 제거할 캐시 키
        """
        cache = self._get_cache_by_type(cache_type)
        if not cache:
            return

        if key in cache:
            del cache[key]
        if key in self._timestamps[cache_type]:
            del self._timestamps[cache_type][key]
        self.stats["size"][cache_type] = len(cache)

    def _evict_lru_item(self, cache_type: str) -> None:
        """
        가장 오래전에 사용된 항목을 제거합니다(LRU 정책).

        Args:
            cache_type: 캐시 유형
        """
        timestamps = self._timestamps.get(cache_type, {})
        if not timestamps:
            return

        # 가장 오래된 타임스탬프를 가진 키 찾기
        oldest_key = min(timestamps.items(), key=lambda x: x[1])[0]
        self._remove(cache_type, oldest_key)
        self.stats["evictions"][cache_type] += 1
        logger.debug(f"LRU 정책으로 캐시 항목 제거: {cache_type}:{oldest_key}")

    def _check_cleanup(self) -> None:
        """
        정기적으로 만료된 캐시 항목을 정리합니다.
        자동으로 호출되어 메모리 관리를 최적화합니다.
        """
        current_time = time.time()

        # 정리 주기 확인
        if current_time - self._last_cleanup < self._cleanup_interval:
            return

        try:
            logger.debug("캐시 정리 시작")
            cleanup_count = 0

            # 각 캐시 유형 정리
            for cache_type in ["session", "chain", "message", "response"]:
                cache = self._get_cache_by_type(cache_type)
                if not cache:
                    continue

                # 만료된 항목 찾기
                expired_keys = []
                for key, timestamp in self._timestamps[cache_type].items():
                    if current_time - timestamp > self.ttl:
                        expired_keys.append(key)

                # 만료된 항목 제거
                for key in expired_keys:
                    self._remove(cache_type, key)
                    cleanup_count += 1
                    self.stats["evictions"][cache_type] += 1

            # 메모리 세트 정리
            if len(self._processed_message_ids) > 10000:
                logger.warning(f"processed_message_ids 크기가 큽니다({len(self._processed_message_ids)}). 초기화합니다.")
                self._processed_message_ids.clear()

            if len(self._processed_inputs) > 1000:
                logger.warning(f"processed_inputs 크기가 큽니다({len(self._processed_inputs)}). 초기화합니다.")
                self._processed_inputs.clear()

            self._last_cleanup = current_time
            if cleanup_count > 0:
                logger.debug(f"캐시 정리 완료: {cleanup_count}개 항목 제거됨")

        except Exception as e:
            logger.error(f"캐시 정리 중 오류: {str(e)}")

    def clear(self, cache_type: Optional[str] = None) -> None:
        """
        지정된 유형 또는An all 캐시를 지웁니다.

        Args:
            cache_type: 지울 캐시 유형 (None이면 모든 캐시)
        """
        if cache_type:
            cache = self._get_cache_by_type(cache_type)
            if cache:
                cache.clear()
                self._timestamps[cache_type].clear()
                self.stats["size"][cache_type] = 0
                self.stats["evictions"][cache_type] += len(cache)
                logger.debug(f"{cache_type} 캐시를 모두 지웠습니다")
        else:
            # 모든 캐시 지우기
            self._session_cache.clear()
            self._chain_cache.clear()
            self._message_cache.clear()
            self._response_cache.clear()
            self._timestamps.clear()

            for cache_type in ["session", "chain", "message", "response"]:
                self.stats["size"][cache_type] = 0

            logger.debug("모든 캐시를 지웠습니다")

    def add_to_processed_set(self, set_type: str, item: Any) -> None:
        """
        처리된 항목 세트에 항목을 추가합니다.

        Args:
            set_type: 세트 유형 ("message_ids" 또는 "inputs")
            item: 추가할 항목
        """
        if set_type == "message_ids":
            self._processed_message_ids.add(item)
        elif set_type == "inputs":
            self._processed_inputs.add(item)

    def remove_from_processed_set(self, set_type: str, item: Any) -> None:
        """
        처리된 항목 세트에서 항목을 제거합니다.

        Args:
            set_type: 세트 유형 ("message_ids" 또는 "inputs")
            item: 제거할 항목
        """
        if set_type == "message_ids" and item in self._processed_message_ids:
            self._processed_message_ids.remove(item)
        elif set_type == "inputs" and item in self._processed_inputs:
            self._processed_inputs.remove(item)

    def is_in_processed_set(self, set_type: str, item: Any) -> bool:
        """
        항목이 처리된 세트에 있는지 확인합니다.

        Args:
            set_type: 세트 유형 ("message_ids" 또는 "inputs")
            item: 확인할 항목

        Returns:
            bool: 항목이 세트에 있으면 True, 아니면 False
        """
        if set_type == "message_ids":
            return item in self._processed_message_ids
        elif set_type == "inputs":
            return item in self._processed_inputs
        return False

    def cleanup_processed_sets(self) -> None:
        """
        처리된 항목 세트를 정리하여 메모리 사용을 최적화합니다.
        """
        self._processed_message_ids.clear()
        self._processed_inputs.clear()
        logger.debug("처리된 항목 세트가 정리되었습니다")

    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """
        캐시 통계를 반환합니다.

        Returns:
            Dict[str, Dict[str, int]]: 캐시 통계 정보
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
