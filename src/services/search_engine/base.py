"""
검색 엔진 기본 인터페이스

검색 엔진 구현을 위한 기본 인터페이스 및 추상 클래스를 제공합니다.
"""

import abc
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document


class SearchEngineBase(abc.ABC):
    """
    검색 엔진 기본 인터페이스

    모든 검색 엔진 구현체가 준수해야 하는 인터페이스를 정의합니다.
    """

    def __init__(self, settings: Any):
        """
        검색 엔진 초기화

        Args:
            settings: 설정 객체
        """
        self.settings = settings

    @abc.abstractmethod
    def search(self, query: str, **kwargs) -> List[Document]:
        """
        동기식 검색을 수행합니다.

        Args:
            query: 검색 쿼리
            **kwargs: 추가 매개변수

        Returns:
            List[Document]: 검색 결과 Document 객체 목록
        """
        pass

    @abc.abstractmethod
    async def search_async(self, query: str, **kwargs) -> List[Document]:
        """
        비동기식 검색을 수행합니다.

        Args:
            query: 검색 쿼리
            **kwargs: 추가 매개변수

        Returns:
            List[Document]: 검색 결과 Document 객체 목록
        """
        pass

    @property
    @abc.abstractmethod
    def engine_name(self) -> str:
        """
        검색 엔진 이름을 반환합니다.

        Returns:
            str: 검색 엔진 이름
        """
        pass

    @property
    def max_results(self) -> int:
        """
        최대 검색 결과 수를 반환합니다.

        Returns:
            int: 최대 검색 결과 수
        """
        return getattr(self.settings.web_search, 'max_results', 10)

    @property
    def region(self) -> str:
        """
        검색 지역 설정을 반환합니다.

        Returns:
            str: 검색 지역 코드
        """
        return getattr(self.settings.web_search, 'region', 'wt-wt')
