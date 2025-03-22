"""
검색 엔진 패키지

웹 검색 기능을 제공하는 모듈과 클래스를 포함합니다.
"""

from src.services.search_engine.base import SearchEngineBase
from src.services.search_engine.duckduckgo_engine import DuckDuckGoEngine
from src.services.search_engine.factory import SearchEngineFactory
from src.services.search_engine.url_processor import URLProcessor
from src.services.search_engine.result_processor import ResultProcessor
from src.services.search_engine.exceptions import (
    SearchEngineError, EngineNotFoundError, SearchTimeoutError, SearchQueryError
)


# 기존 코드와의 호환성을 위한 SearchEngine 클래스
class SearchEngine:
    """
    기존 SearchEngine 클래스의 호환성 래퍼

    기존 SearchEngine 클래스의 API를 유지하면서 내부적으로
    새로운 아키텍처를 사용합니다.
    """

    def __init__(self, settings):
        """
        SearchEngine 초기화

        Args:
            settings: 설정 객체
        """
        self.settings = settings
        self._engine = SearchEngineFactory.create_default_engine(settings)
        self._url_processor = URLProcessor()

    def websearch_duckduckgo(self, query: str, sleep_duration: float = 1.0) -> list:
        """
        DuckDuckGo 웹 검색을 수행합니다.

        Args:
            query: 검색 쿼리
            sleep_duration: 검색 후 대기 시간(초)

        Returns:
            list: 검색 결과 Document 객체 목록
        """
        return self._engine.search(query, sleep_duration=sleep_duration)

    async def websearch_duckduckgo_async(self, query: str, sleep_duration: float = 1.0) -> list:
        """
        DuckDuckGo 웹 검색을 비동기로 수행합니다.

        Args:
            query: 검색 쿼리
            sleep_duration: 검색 후 대기 시간(초)

        Returns:
            list: 검색 결과 Document 객체 목록
        """
        return await self._engine.search_async(query, sleep_duration=sleep_duration)

    def replace_urls_with_links(self, query_answer: str) -> str:
        """
        텍스트에서 URL을 하이퍼링크로 변환합니다.

        Args:
            query_answer: 변환할 텍스트

        Returns:
            str: URL이 하이퍼링크로 변환된 텍스트
        """
        return self._url_processor.convert_urls_to_links(query_answer)


__all__ = [
    'SearchEngine',
    'SearchEngineBase',
    'DuckDuckGoEngine',
    'SearchEngineFactory',
    'URLProcessor',
    'ResultProcessor',
    'SearchEngineError',
    'EngineNotFoundError',
    'SearchTimeoutError',
    'SearchQueryError'
]
