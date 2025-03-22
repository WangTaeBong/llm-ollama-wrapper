"""
패턴 필터 모듈

패턴 기반 쿼리 필터링 기능을 제공합니다.
"""

import logging
import re
from typing import List, Set, Dict, Any

# 로거 설정
logger = logging.getLogger(__name__)


class PatternFilter:
    """
    패턴 필터 클래스

    패턴 기반 쿼리 필터링 및 매칭 기능을 제공합니다.
    """

    def __init__(self, settings, query_check_json_dict=None):
        """
        패턴 필터 초기화

        Args:
            settings: 설정 객체
            query_check_json_dict: 쿼리 패턴 사전 (선택 사항)
        """
        self.settings = settings
        self.query_check_json_dict = query_check_json_dict

        # 패턴 캐시 초기화
        self._pattern_cache: Dict[str, Set[str]] = {}

        logger.debug("패턴 필터가 초기화되었습니다")

    def get_pattern_set(self, lang: str, dict_key: str) -> Set[str]:
        """
        지정된 언어와 사전 키에 대한 패턴 집합을 반환합니다.

        Args:
            lang: 언어 코드
            dict_key: 사전 키

        Returns:
            Set[str]: 패턴 집합 또는 빈 집합
        """
        cache_key = f"{lang}:{dict_key}"

        # 캐시에서 가져오기
        if cache_key in self._pattern_cache:
            return self._pattern_cache[cache_key]

        # 패턴 목록 가져오기
        patterns = self.query_check_json_dict.get_dict_data(lang, dict_key)
        if not patterns:
            self._pattern_cache[cache_key] = set()
            return set()

        # 패턴 집합 생성 및 캐시
        pattern_set = set(patterns)
        self._pattern_cache[cache_key] = pattern_set

        return pattern_set

    def match_pattern(self, query: str, lang: str, dict_keys: List[str]) -> Dict[str, Any]:
        """
        쿼리가 지정된 패턴과 일치하는지 확인합니다.

        Args:
            query: 확인할 쿼리
            lang: 언어 코드
            dict_keys: 확인할 사전 키 목록

        Returns:
            Dict[str, Any]: 매칭 결과 (matched: 일치 여부, key: 일치한 키, lang: 일치한 언어)
        """
        for dict_key in dict_keys:
            pattern_set = self.get_pattern_set(lang, dict_key)

            if query in pattern_set:
                return {
                    "matched": True,
                    "key": dict_key,
                    "lang": lang
                }

        return {"matched": False}

    def clear_cache(self) -> None:
        """패턴 캐시를 지웁니다."""
        self._pattern_cache.clear()
        logger.debug("패턴 캐시를 지웠습니다")
        