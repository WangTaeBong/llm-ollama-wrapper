"""
쿼리 처리기 기본 인터페이스 모듈
==========================

다양한 쿼리 처리기에 대한 표준 인터페이스를 정의합니다.
"""

import abc
import logging
from typing import Optional

from src.schema.chat_req import ChatRequest

# 로거 설정
logger = logging.getLogger(__name__)


class QueryProcessorBase(abc.ABC):
    """
    쿼리 처리기 기본 인터페이스

    모든 쿼리 처리기 구현은 이 클래스를 상속받아 구현해야 합니다.
    """

    def __init__(self, settings: Any, query_check_dict: Any = None):
        """
        쿼리 처리기 초기화

        Args:
            settings: 설정 객체
            query_check_dict: 쿼리 체크 딕셔너리 (선택 사항)
        """
        self.settings = settings
        self.query_check_dict = query_check_dict

    @abc.abstractmethod
    async def initialize(self) -> bool:
        """
        쿼리 처리기 초기화

        Returns:
            bool: 초기화 성공 여부
        """
        pass

    @abc.abstractmethod
    def clean_query(self, query: str) -> str:
        """
        쿼리 정제

        Args:
            query: 원본 쿼리

        Returns:
            str: 정제된 쿼리
        """
        pass

    @abc.abstractmethod
    def filter_query(self, query: str) -> str:
        """
        쿼리 필터링

        Args:
            query: 원본 쿼리

        Returns:
            str: 필터링된 쿼리
        """
        pass

    @abc.abstractmethod
    async def check_query_sentence(self, request: ChatRequest) -> Optional[str]:
        """
        쿼리 문장 확인

        Args:
            request: 채팅 요청 객체

        Returns:
            Optional[str]: 특별 응답 또는 None
        """
        pass

    @abc.abstractmethod
    async def construct_faq_query(self, request: ChatRequest) -> str:
        """
        FAQ 쿼리 구성

        Args:
            request: 채팅 요청 객체

        Returns:
            str: 구성된 FAQ 쿼리 또는 원본 쿼리
        """
        pass
