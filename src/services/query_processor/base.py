"""
쿼리 프로세서 기본 인터페이스

쿼리 프로세서 구현을 위한 기본 인터페이스 및 추상 클래스를 제공합니다.
"""

import abc
from typing import Optional


class QueryProcessorBase(abc.ABC):
    """
    쿼리 프로세서 기본 인터페이스

    모든 쿼리 프로세서 구현체가 준수해야 하는 인터페이스를 정의합니다.
    """

    def __init__(self, settings, query_check_json_dict=None):
        """
        쿼리 프로세서 초기화

        Args:
            settings: 설정 객체
            query_check_json_dict: 쿼리 패턴 사전 (선택 사항)
        """
        self.settings = settings
        self.query_check_json_dict = query_check_json_dict

    @abc.abstractmethod
    def clean_query(self, query: str) -> str:
        """
        쿼리를 정제합니다.

        Args:
            query: 정제할 쿼리

        Returns:
            str: 정제된 쿼리
        """
        pass

    @abc.abstractmethod
    def filter_query(self, query: str) -> str:
        """
        쿼리를 필터링합니다.

        Args:
            query: 필터링할 쿼리

        Returns:
            str: 필터링된 쿼리
        """
        pass

    @property
    def processor_name(self) -> str:
        """
        프로세서 이름을 반환합니다.

        Returns:
            str: 프로세서 이름
        """
        return self.__class__.__name__
