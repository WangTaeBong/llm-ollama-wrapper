"""
FAQ 쿼리 프로세서 모듈

FAQ 유형에 최적화된 쿼리 구성 기능을 제공합니다.
"""

import logging
from typing import Set, List

from src.services.query_processor.base import QueryProcessorBase

# 로거 설정
logger = logging.getLogger(__name__)


class FAQQueryProcessor(QueryProcessorBase):
    """
    FAQ 쿼리 프로세서 클래스

    FAQ 유형에 최적화된 쿼리 구성 기능을 제공합니다.
    """

    def __init__(self, settings, query_check_json_dict=None):
        """
        FAQ 쿼리 프로세서 초기화

        Args:
            settings: 설정 객체
            query_check_json_dict: 쿼리 패턴 사전 (선택 사항)
        """
        super().__init__(settings, query_check_json_dict)

        # 자주 사용되는 데이터 캐싱
        self._faq_category_rag_targets: List[str] = self.settings.prompt.faq_type.split(',')
        self._excluded_categories: Set[str] = {"담당자 메일 문의", "AI 직접 질문", "챗봇 문의"}

        logger.debug("FAQ 쿼리 프로세서가 초기화되었습니다")

    def clean_query(self, query: str) -> str:
        """
        FAQ 컨텍스트에 맞게 쿼리를 정제합니다.

        Args:
            query (str): 정제할 원본 쿼리

        Returns:
            str: 정제된 쿼리
        """
        if not query:
            return ""
        # FAQ에 맞는 정제 로직 적용
        return query.strip()

    def filter_query(self, query: str) -> str:
        """
        FAQ 컨텍스트에 맞게 쿼리를 필터링합니다.
        FAQ는 원본 텍스트를 최대한 보존합니다.

        Args:
            query (str): 필터링할 원본 쿼리

        Returns:
            str: 필터링된 쿼리 (FAQ의 경우 대부분 원본 유지)
        """
        return query

    def construct_faq_query(self, request) -> str:
        """
        FAQ 카테고리 기반으로 최적화된 LLM 쿼리를 생성합니다.

        Args:
            request: 채팅 요청 객체

        Returns:
            str: 구성된 FAQ 쿼리 또는 원본 쿼리
        """
        # RAG 시스템 정보가 FAQ 유형에 해당하는지 확인
        if request.meta.rag_sys_info not in self._faq_category_rag_targets:
            return request.chat.user

        # 쿼리 구성
        query_parts = []

        # 카테고리별 처리
        category_suffixes = [
            (request.chat.category1, " 카테고리에 속합니다."),
            (request.chat.category2, " 카테고리에 속합니다."),
            (request.chat.category3, " 정보입니다.")
        ]

        for category, suffix in category_suffixes:
            if category and category not in self._excluded_categories:
                query_parts.append(f"{category}{suffix}")

        # 원본 쿼리와 결합
        if query_parts:
            return " ".join(query_parts) + " " + request.chat.user

        return request.chat.user
