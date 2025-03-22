"""
패턴 쿼리 프로세서 모듈

쿼리 패턴 감지 및 응답 생성 기능을 제공합니다.
"""

import logging
import random
import re
from typing import Optional, Set, Dict, Any

from src.services.query_processor.base import QueryProcessorBase
from src.services.query_processor.cache_manager import QueryCacheManager

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

        # 캐시 관리자 초기화
        self.cache_manager = QueryCacheManager()

        # 표준 정제 패턴
        self._clean_query_pattern = re.compile(r'[~!@#$%^&*()=+\[\]{}:?,<>/\-_.]')

        logger.debug("패턴 쿼리 프로세서가 초기화되었습니다")

    def clean_query(self, query: str) -> str:
        """
        사용자 입력 쿼리에서 불필요한 특수 문자와 기호를 제거합니다.

        Args:
            query (str): 정제할 원본 쿼리

        Returns:
            str: 정제된 쿼리
        """
        if not query:
            return ""
        return self._clean_query_pattern.sub('', query.lower())

    def filter_query(self, query: str) -> str:
        """
        패턴 매칭을 위한 최소한의 필터링만 적용합니다.

        Args:
            query (str): 필터링할 원본 쿼리

        Returns:
            str: 필터링된 쿼리
        """
        return query.lower() if query else ""

    def check_query_sentence(self, request) -> Optional[str]:
        """
        사용자 쿼리를 미리 정의된 응답 패턴과 비교하여 적절한 응답을 생성합니다.
        인사말, 종료 등의 특정 패턴을 인식하여 적절한 응답을 반환합니다.

        Args:
            request: 채팅 요청 객체

        Returns:
            Optional[str]: 응답 문자열(일치하는 패턴이 있는 경우), None(없는 경우)
        """
        # 캐시 키 생성
        cache_key = f"pattern_check:{request.meta.session_id}:{request.chat.user}"
        cached_response = self.cache_manager.get(cache_key)
        if cached_response is not None:
            logger.debug(f"[{request.meta.session_id}] 패턴 캐시 사용: {cache_key}")
            return cached_response

        # 설정 가져오기
        query_lang_key_list = self.settings.lm_check.query_lang_key.split(',')
        query_dict_key_list = self.settings.lm_check.query_dict_key.split(',')

        if not query_lang_key_list or not query_dict_key_list:
            return None

        # 사용자 언어 코드 (ko, en, jp, cn 등)
        user_lang = request.chat.lang

        # 쿼리 정제
        raw_query = self.clean_query(request.chat.user)

        # 너무 짧거나 숫자로만 구성된 쿼리 처리
        if len(raw_query) < 2 or raw_query.isdigit():
            farewells_msgs = self.query_check_json_dict.get_dict_data(user_lang, "farewells_msg")
            if farewells_msgs:
                response = random.choice(farewells_msgs)
                self.cache_manager.set(cache_key, response)
                return response
            return None

        try:
            # 캐시 초기화 - 각 언어 코드에 대한 패턴 사전
            pattern_cache = {}

            # 1. 먼저 사용자 언어 코드와 일치하는 패턴 확인 (최적화)
            if user_lang in query_lang_key_list:
                for data_dict in query_dict_key_list:
                    # 패턴 목록 가져오기 (인사말, 종료 등)
                    patterns = self.query_check_json_dict.get_dict_data(user_lang, data_dict)
                    if not patterns:
                        continue

                    # 패턴을 집합으로 변환하고 캐싱 (성능 최적화)
                    pattern_set = set(patterns)
                    pattern_cache[(user_lang, data_dict)] = pattern_set

                    # 현재 쿼리가 패턴 집합에 있는지 확인
                    if raw_query in pattern_set:
                        # 응답 키 구성 (예: greetings → greetings_msg)
                        response_key = f"{data_dict}_msg"
                        # 일치하는 언어에 대한 응답 메시지 가져오기
                        response_messages = self.query_check_json_dict.get_dict_data(user_lang, response_key)
                        if response_messages:
                            response = random.choice(response_messages)
                            self.cache_manager.set(cache_key, response)
                            return response

            # 2. 사용자 언어에서 찾지 못한 경우 다른 언어 코드 확인
            for chat_lang in query_lang_key_list:
                # 이미 확인한 사용자 언어 건너뛰기
                if chat_lang == user_lang:
                    continue

                for data_dict in query_dict_key_list:
                    # 캐시에서 가져오거나 없으면 가져와서 캐싱
                    if (chat_lang, data_dict) in pattern_cache:
                        pattern_set = pattern_cache[(chat_lang, data_dict)]
                    else:
                        patterns = self.query_check_json_dict.get_dict_data(chat_lang, data_dict)
                        if not patterns:
                            continue
                        pattern_set = set(patterns)
                        pattern_cache[(chat_lang, data_dict)] = pattern_set

                    # 패턴 매칭
                    if raw_query in pattern_set:
                        # 다른 언어에서 패턴을 찾았지만 사용자 언어로 응답
                        response_key = f"{data_dict}_msg"
                        response_messages = self.query_check_json_dict.get_dict_data(user_lang, response_key)
                        if response_messages:
                            response = random.choice(response_messages)
                            self.cache_manager.set(cache_key, response)
                            return response

            # 패턴이 일치하지 않으면 None 반환
            return None

        except Exception as e:
            logger.warning(f"[{request.meta.session_id}] check_query_sentence 오류: {e}")
            # 오류 시 None 반환
            return None
