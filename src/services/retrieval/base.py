"""
검색 기능의 기본 인터페이스 모듈

모든 검색 관련 구현체가 따라야 할 기본 인터페이스와 추상 클래스를 제공합니다.
"""

import abc
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun


class DataProviderBase(abc.ABC):
    """
    데이터 제공자 인터페이스

    문서 데이터를 검색하는 다양한 소스의 인터페이스를 정의합니다.
    """

    @abc.abstractmethod
    async def fetch_documents(self, query: str, **kwargs) -> List[Document]:
        """
        쿼리에 대한 문서를 비동기적으로 가져옵니다.

        Args:
            query: 검색 쿼리 문자열
            **kwargs: 추가 매개변수

        Returns:
            List[Document]: 검색된 문서 목록
        """
        pass


class CacheManagerBase(abc.ABC):
    """
    캐시 관리 인터페이스

    검색 결과와 기타 데이터의 캐싱을 위한 인터페이스입니다.
    """

    @abc.abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """
        캐시에서 값을 가져옵니다.

        Args:
            key: 캐시 키

        Returns:
            Optional[Any]: 캐시된 값 또는 None (캐시 미스 또는 만료)
        """
        pass

    @abc.abstractmethod
    def set(self, key: str, value: Any) -> None:
        """
        캐시에 항목을 저장합니다.

        Args:
            key: 캐시 키
            value: 저장할 값
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def create_key(data: Any) -> str:
        """
        데이터로부터 캐시 키를 생성합니다.

        Args:
            data: 키를 생성할 데이터

        Returns:
            str: 생성된 캐시 키
        """
        pass


class DocumentStoreBase(abc.ABC):
    """
    문서 저장소 인터페이스

    Document 객체의 저장 및 검색을 위한 인터페이스입니다.
    """

    @abc.abstractmethod
    def add(self, document: Document) -> bool:
        """
        문서를 저장소에 추가합니다.

        Args:
            document: 추가할 Document 객체

        Returns:
            bool: 추가 성공 시 True, 중복인 경우 False
        """
        pass

    @abc.abstractmethod
    def add_batch(self, documents: List[Document]) -> int:
        """
        문서 배치를 저장소에 추가합니다.

        Args:
            documents: 추가할 Document 객체 목록

        Returns:
            int: 성공적으로 추가된 문서 수
        """
        pass

    @abc.abstractmethod
    def get_all(self) -> List[Document]:
        """
        모든 저장된 문서를 가져옵니다.

        Returns:
            List[Document]: 저장된 모든 문서 목록
        """
        pass


class RetrieverBase(abc.ABC):
    """
    검색기 기본 인터페이스

    문서 검색 구현체의 기본 인터페이스를 정의합니다.
    """

    @abc.abstractmethod
    async def ainvoke(self,
                      query: str,
                      run_manager: Optional[CallbackManagerForRetrieverRun] = None,
                      **kwargs) -> List[Document]:
        """
        비동기적으로 문서를 검색합니다.

        Args:
            query: 검색 쿼리
            run_manager: 콜백 관리자(선택 사항)
            **kwargs: 추가 매개변수

        Returns:
            List[Document]: 검색된 문서 목록
        """
        pass
