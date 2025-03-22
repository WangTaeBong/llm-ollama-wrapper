"""
DuckDuckGo 검색 엔진 구현

DuckDuckGo API를 사용한 검색 엔진 구현을 제공합니다.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional

from duckduckgo_search import DDGS
from langchain_core.documents import Document
from tenacity import retry, stop_after_attempt, wait_exponential

from src.services.search_engine.base import SearchEngineBase
from src.services.search_engine.result_processor import ResultProcessor

# 로거 설정
logger = logging.getLogger(__name__)


class DuckDuckGoEngine(SearchEngineBase):
    """
    DuckDuckGo 검색 엔진 구현

    DuckDuckGo API를 사용하여 웹 검색을 수행합니다.
    """

    def __init__(self, settings: Any):
        """
        DuckDuckGo 검색 엔진 초기화

        Args:
            settings: 설정 객체
        """
        super().__init__(settings)
        self.result_processor = ResultProcessor()
        logger.debug("DuckDuckGo 검색 엔진이 초기화되었습니다")

    @property
    def engine_name(self) -> str:
        """
        검색 엔진 이름을 반환합니다.

        Returns:
            str: 검색 엔진 이름
        """
        return "DuckDuckGo"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
    def search(self, query: str, sleep_duration: float = 1.0, **kwargs) -> List[Document]:
        """
        DuckDuckGo를 사용하여 동기식 검색을 수행합니다.

        Args:
            query: 검색 쿼리
            sleep_duration: 검색 후 대기 시간(초)
            **kwargs: 추가 매개변수

        Returns:
            List[Document]: 검색 결과 Document 객체 목록
        """
        if not query or not query.strip():
            logger.warning("빈 검색 쿼리가 제공되었습니다")
            return []

        start_time = time.time()

        try:
            with DDGS() as ddgs:
                # 검색 설정 가져오기
                region = kwargs.get('region', self.region)
                max_results = kwargs.get('max_results', self.max_results)

                # 검색 실행
                results = list(ddgs.text(query, region=region, max_results=max_results))

                # 결과를 Document 객체로 변환
                documents = self.result_processor.convert_to_documents(results)

            # 요청 간 간격 유지
            if sleep_duration > 0:
                time.sleep(sleep_duration)

            # 성능 로깅
            elapsed_time = time.time() - start_time
            if documents:
                logger.info(
                    f"검색 '{query[:30]}...'에서 {len(documents)}개 결과 반환, "
                    f"처리 시간: {elapsed_time:.2f}초"
                )
            else:
                logger.warning(
                    f"검색 '{query[:30]}...'에서 결과 없음 (처리 시간: {elapsed_time:.2f}초)"
                )

            return documents

        except Exception as e:
            logger.error(f"DuckDuckGo 검색 중 오류 발생 (query='{query[:30]}...'): {str(e)}")
            return []

    async def search_async(self, query: str, sleep_duration: float = 1.0, **kwargs) -> List[Document]:
        """
        DuckDuckGo를 사용하여 비동기식 검색을 수행합니다.

        Args:
            query: 검색 쿼리
            sleep_duration: 검색 후 대기 시간(초)
            **kwargs: 추가 매개변수

        Returns:
            List[Document]: 검색 결과 Document 객체 목록
        """
        if not query or not query.strip():
            return []

        start_time = time.time()

        try:
            # DuckDuckGo 검색은 동기식이지만 비동기 컨텍스트에서 사용 가능하도록 별도 스레드에서 실행
            results = await asyncio.to_thread(
                self._perform_search,
                query,
                kwargs.get('region', self.region),
                kwargs.get('max_results', self.max_results)
            )

            # 결과를 Document 객체로 변환
            documents = self.result_processor.convert_to_documents(results)

            # 비동기 대기
            if sleep_duration > 0:
                await asyncio.sleep(sleep_duration)

            # 성능 로깅
            elapsed_time = time.time() - start_time
            if documents:
                logger.info(
                    f"비동기 검색 '{query[:30]}...'에서 {len(documents)}개 결과 반환, "
                    f"처리 시간: {elapsed_time:.2f}초"
                )
            else:
                logger.warning(
                    f"비동기 검색 '{query[:30]}...'에서 결과 없음 "
                    f"(처리 시간: {elapsed_time:.2f}초)"
                )

            return documents

        except Exception as e:
            logger.error(f"비동기 DuckDuckGo 검색 중 오류 발생 (query='{query[:30]}...'): {str(e)}")
            return []

    @classmethod
    def _perform_search(cls, query: str, region: str, max_results: int) -> List[Dict[str, Any]]:
        """
        실제 DuckDuckGo 검색을 수행합니다.

        Args:
            query: 검색 쿼리
            region: 검색 지역
            max_results: 최대 결과 수

        Returns:
            List[Dict[str, Any]]: 검색 결과 목록
        """
        with DDGS() as ddgs:
            try:
                # 임시로 디버그 로그 비활성화
                original_level = logging.getLogger().level
                logging.getLogger().setLevel(logging.WARNING)

                try:
                    return list(ddgs.text(query, region=region, max_results=max_results))
                finally:
                    # 원래 로깅 레벨 복원
                    logging.getLogger().setLevel(original_level)
            except Exception as e:
                logger.error(f"DuckDuckGo API 호출 중 오류: {e}")
                return []
