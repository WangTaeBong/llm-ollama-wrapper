"""
쿼리 처리 모듈
===========

사용자 쿼리를 처리하고 최적화하는 기능을 제공합니다.

기능:
- 쿼리 정제 및 필터링
- 패턴 감지 및 특수 응답 생성
- FAQ 쿼리 최적화
"""

import logging
import random
import re
from typing import Optional, Set, List

from src.services.query.base import QueryProcessorBase
from src.services.query.factory import QueryProcessorFactory
from src.schema.chat_req import ChatRequest

# 로거 설정
logger = logging.getLogger(__name__)


class StandardQueryProcessor(QueryProcessorBase):
    """
    표준 쿼리 처리기 구현

    쿼리 정제, 필터링, 패턴 감지 등 기본 쿼리 처리를 담당합니다.
    """

    def __init__(self, settings, query_check_dict):
        """
        표준 쿼리 처리기 초기화

        Args:
            settings: 설정 객체
            query_check_dict: 쿼리 체크 딕셔너리
        """
        super().__init__(settings)
        self.query_check_dict = query_check_dict

        # 컴파일된 정규 표현식 패턴 (성능 최적화)
        self._clean_query_pattern = re.compile(r'[~!@#$%^&*()=+\[\]{}:?,<>/\-_.]')
        self._ko_jamo_pattern = re.compile(r'([ㄱ-ㅎㅏ-ㅣ\s]+)')
        self._arabia_num_pattern = re.compile(r'([0-9]+)')
        self._wild_char_pattern = re.compile(r'([^\w\s]+)')

        # 캐싱된 설정 값
        self._faq_category_rag_targets = self.settings.prompt.faq_type.split(',')
        self._excluded_categories = {"담당자 메일 문의", "AI 직접 질문", "챗봇 문의"}

    @classmethod
    async def initialize(cls) -> bool:
        """
        처리기 초기화

        Returns:
            bool: 초기화 성공 여부
        """
        # 초기화 성공
        return True

    def clean_query(self, query: str) -> str:
        """
        쿼리 정제 - 특수 문자 제거

        Args:
            query: 원본 쿼리

        Returns:
            str: 정제된 쿼리
        """
        if not query:
            return ""
        return self._clean_query_pattern.sub('', query.lower())

    def filter_query(self, query: str) -> str:
        """
        필터링 적용 - 불필요한 문자 제거

        Args:
            query: 원본 쿼리

        Returns:
            str: 필터링된 쿼리
        """
        if not query:
            return ""

        try:
            # 설정에 따라 필터링 패턴 적용
            if self.settings.query_filter.ko_jamo:
                query = self._ko_jamo_pattern.sub('', query)
            if self.settings.query_filter.arabia_num:
                query = self._arabia_num_pattern.sub('', query)
            if self.settings.query_filter.wild_char:
                query = self._wild_char_pattern.sub('', query)
            return query
        except Exception as e:
            logger.warning(f"쿼리 필터링 중 오류: {e}")
            return query

    async def check_query_sentence(self, request: ChatRequest) -> Optional[str]:
        """
        쿼리 문장 패턴 체크 - 인사말 등 특수 패턴 감지

        Args:
            request: 채팅 요청 객체

        Returns:
            Optional[str]: 감지된 패턴에 대한 응답 또는 None
        """
        # 설정 가져오기
        query_lang_key_list = self.settings.lm_check.query_lang_key.split(',')
        query_dict_key_list = self.settings.lm_check.query_dict_key.split(',')

        if not query_lang_key_list or not query_dict_key_list:
            return None

        # 사용자 언어 코드 (ko, en, jp, cn 등)
        user_lang = request.chat.lang

        # 쿼리 정제
        raw_query = self.clean_query(request.chat.user)

        # 짧거나 숫자로만 된 쿼리 처리
        if len(raw_query) < 2 or raw_query.isdigit():
            farewells_msgs = self.query_check_dict.get_dict_data(user_lang, "farewells_msg")
            if farewells_msgs:
                return random.choice(farewells_msgs)
            return None

        # 패턴 매칭 - 모든 언어 키 및 사전 키에 대해 확인
        try:
            # 패턴 캐싱
            pattern_cache = {}

            # 1. 먼저 사용자 언어 코드와 일치하는 패턴 확인 (최적화)
            if user_lang in query_lang_key_list:
                for data_dict in query_dict_key_list:
                    # 패턴 목록 가져오기
                    patterns = self.query_check_dict.get_dict_data(user_lang, data_dict)
                    if not patterns:
                        continue

                    # 패턴을 집합으로 변환하여 캐싱 (성능 최적화)
                    pattern_set = set(patterns)
                    pattern_cache[(user_lang, data_dict)] = pattern_set

                    # 현재 쿼리가 패턴 집합에 존재하는지 확인
                    if raw_query in pattern_set:
                        # 응답 키 구성 (예: greetings → greetings_msg)
                        response_key = f"{data_dict}_msg"
                        # 일치하는 언어의 응답 메시지 가져오기
                        response_messages = self.query_check_dict.get_dict_data(user_lang, response_key)
                        if response_messages:
                            return random.choice(response_messages)

            # 2. 사용자 언어에서 찾지 못한 경우 다른 언어 코드 확인
            for chat_lang in query_lang_key_list:
                # 이미 확인한 사용자 언어 건너뛰기
                if chat_lang == user_lang:
                    continue

                for data_dict in query_dict_key_list:
                    # 캐시에서 재사용하거나 새로 가져와서 캐싱
                    if (chat_lang, data_dict) in pattern_cache:
                        pattern_set = pattern_cache[(chat_lang, data_dict)]
                    else:
                        patterns = self.query_check_dict.get_dict_data(chat_lang, data_dict)
                        if not patterns:
                            continue
                        pattern_set = set(patterns)
                        pattern_cache[(chat_lang, data_dict)] = pattern_set

                    # 패턴 매칭
                    if raw_query in pattern_set:
                        # 다른 언어에서 패턴 발견, 사용자 언어로 응답 제공
                        response_key = f"{data_dict}_msg"
                        response_messages = self.query_check_dict.get_dict_data(user_lang, response_key)
                        if response_messages:
                            return random.choice(response_messages)

            # 패턴이 일치하지 않는 경우 None 반환
            return None

        except Exception as e:
            logger.warning(f"[{request.meta.session_id}] 쿼리 문장 체크 중 오류: {e}")
            # 오류 발생 시 None 반환
            return None

    async def construct_faq_query(self, request: ChatRequest) -> str:
        """
        FAQ 쿼리 구성 - 카테고리 기반 쿼리 최적화

        Args:
            request: 채팅 요청 객체

        Returns:
            str: 구성된 FAQ 쿼리 또는 원본 쿼리
        """
        # RAG 시스템 정보가 FAQ 유형인지 확인
        if request.meta.rag_sys_info not in self._faq_category_rag_targets:
            return request.chat.user

        # 쿼리 구성 부분
        query_parts = []

        # 카테고리별 처리
        category_suffixes = [
            (request.chat.category1, " belongs to this category."),
            (request.chat.category2, " belongs to this category."),
            (request.chat.category3, " is this information.")
        ]

        for category, suffix in category_suffixes:
            if category and category not in self._excluded_categories:
                query_parts.append(f"{category}{suffix}")

        # 원본 쿼리와 결합
        if query_parts:
            return " ".join(query_parts) + " " + request.chat.user

        return request.chat.user


# 클래스 등록
QueryProcessorFactory.register_processor("standard", StandardQueryProcessor)
