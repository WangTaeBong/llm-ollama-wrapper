"""
패턴 쿼리 프로세서 모듈

쿼리 패턴 감지 및 응답 생성 기능을 제공합니다.
"""

import logging
import random
from typing import Optional, Dict, Set

from src.services.query_processor.base import QueryProcessorBase
from src.services.query_processor.cache_manager import QueryCache

# 로거 설정
logger = logging.getLogger(__name__)


class PatternQueryProcessor(QueryProcessorBase):
    """
    패턴 쿼리 프로세서 클래스

    쿼리 패턴 감지 및 응답 생성 기능을 제공합니다.
    """

    def __init__(self, settings, query_check_json_dict=None):
        """
        패턴 쿼리 프로세서 초기화

        Args:
            settings: 설정 객체
            query_check_json_dict: 쿼리 패턴 사전
        """
        super().__init__(settings, query_check_json_dict)

        # 캐시 인스턴스 가져오기
        self.cache = QueryCache.get_instance()

        # 패턴 캐시 초기화
        self._pattern_cache: Dict[str, Set[str]] = {}

        # logger.debug("패턴 쿼리 프로세서가 초기화되었습니다")

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

    def check_query_sentence(self, request) -> Optional[str]:
        """
        사용자 쿼리를 미리 정의된 응답 패턴과 비교하여 적절한 응답을 생성합니다.

        Args:
            request: 채팅 요청 객체

        Returns:
            Optional[str]: 응답 문자열(일치하는 패턴이 있는 경우), None(없는 경우)
        """
        # 캐시 키 생성
        cache_key = f"pattern:{request.meta.session_id}:{request.chat.user}"

        # 캐시에서 결과 확인
        cached_response = self.cache.get(cache_key)
        if cached_response is not None:
            logger.debug(f"[{request.meta.session_id}] 패턴 캐시 사용: {cache_key}")
            return cached_response

        user_lang = request.chat.lang
        raw_query = self.clean_query(request.chat.user)

        # 너무 짧거나 숫자로만 구성된 쿼리 처리
        if len(raw_query) < 2 or raw_query.isdigit():
            farewells_msgs = self.query_check_json_dict.get_dict_data(user_lang, "farewells_msg")
            if farewells_msgs:
                response = random.choice(farewells_msgs)
                self.cache.set(cache_key, response)
                return response
            return None

        try:
            # 설정 가져오기
            query_lang_key_list = self.settings.lm_check.query_lang_key.split(',')
            query_dict_key_list = self.settings.lm_check.query_dict_key.split(',')

            # 패턴 매칭 - 사용자 언어 먼저 확인
            if user_lang in query_lang_key_list:
                result = self._match_patterns_for_language(raw_query, user_lang, query_dict_key_list, user_lang)
                if result:
                    self.cache.set(cache_key, result)
                    return result

            # 다른 언어 확인
            for lang in query_lang_key_list:
                if lang == user_lang:
                    continue

                result = self._match_patterns_for_language(raw_query, lang, query_dict_key_list, user_lang)
                if result:
                    self.cache.set(cache_key, result)
                    return result

            return None

        except Exception as e:
            logger.warning(f"[{request.meta.session_id}] check_query_sentence 오류: {e}")
            return None

    def _match_patterns_for_language(self, query: str, pattern_lang: str,
                                     dict_keys: list, response_lang: str) -> Optional[str]:
        """
        특정 언어의 패턴과 쿼리를 매칭하고 응답을 생성합니다.

        Args:
            query: 확인할 쿼리
            pattern_lang: 패턴 언어
            dict_keys: 확인할 사전 키 목록
            response_lang: 응답 생성을 위한 언어

        Returns:
            Optional[str]: 매칭된 응답 또는 None
        """
        for dict_key in dict_keys:
            pattern_set = self.get_pattern_set(pattern_lang, dict_key)

            if query in pattern_set:
                # 응답 키 구성 (예: greetings → greetings_msg)
                response_key = f"{dict_key}_msg"
                response_messages = self.query_check_json_dict.get_dict_data(response_lang, response_key)

                if response_messages:
                    return random.choice(response_messages)

        return None
