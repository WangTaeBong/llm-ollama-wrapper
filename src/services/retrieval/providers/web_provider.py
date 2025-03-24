"""
웹 검색 제공자 모듈

웹 검색을 통한 문서 검색 기능을 제공합니다.
"""

import asyncio
import logging
import time
from typing import List, Any, Optional

from langchain_core.documents import Document

from src.services.retrieval.base import DataProviderBase, CacheManagerBase

# 로거 설정
logger = logging.getLogger(__name__)


class WebSearchProvider(DataProviderBase):
    """
    웹 검색 제공자 클래스

    웹 검색을 통해 문서를 검색하는 기능을 제공합니다.
    """

    def __init__(self, settings: Any, cache_manager: CacheManagerBase, timeout: float = 5.0):
        """
        웹 검색 제공자 초기화

        Args:
            settings: 애플리케이션 설정
            cache_manager: 캐시 관리자
            timeout: 웹 검색 제한 시간(초)
        """
        self.settings = settings
        self.cache_manager = cache_manager
        self.timeout = timeout

        # 필요한 구성 요소 초기화
        self._initialize_components()

    def _initialize_components(self):
        """검색 엔진 및 기타 필요한 구성 요소 초기화"""
        from src.common.query_check_dict import QueryCheckDict
        from src.services.response_generator import ResponseGenerator
        from src.services.search_engine import SearchEngine

        query_check_dict = QueryCheckDict(self.settings.prompt.llm_prompt_path)
        self.response_generator = ResponseGenerator(self.settings, query_check_dict)
        self.search_engine = SearchEngine(self.settings)

    async def fetch_documents(self,
                              query: str,
                              rag_sys_info: str = None,
                              session_id: str = None,
                              **kwargs) -> List[Document]:
        """
        웹 검색을 수행하고 문서를 가져옵니다.

        Args:
            query: 검색 쿼리
            rag_sys_info: RAG 시스템 정보
            session_id: 세션 ID (로깅용)
            **kwargs: 추가 매개변수

        Returns:
            List[Document]: 검색된 문서 목록
        """
        # 추가 매개변수에서 sleep_duration 추출
        sleep_duration = kwargs.get('sleep_duration', 1.0)

        # 웹 검색이 비활성화되었거나 FAQ 챗봇인 경우 빈 목록 반환
        if (not self.settings.web_search.use_flag or
                (rag_sys_info and self.response_generator.is_faq_type_chatbot(rag_sys_info)) or
                self.response_generator.is_voc_type_chatbot(rag_sys_info)):
            return []

        # 캐시 키 생성
        cache_key = self.cache_manager.create_key(f"websearch_{query}_{rag_sys_info}")

        # 캐시 확인
        cached_results = self.cache_manager.get(cache_key)
        if cached_results is not None:
            logger.debug(f"[{session_id}] 웹 검색 캐시 적중: {query}")
            return cached_results

        # 비동기적으로 웹 검색 수행
        try:
            start_time = time.time()
            logger.debug(f"[{session_id}] 웹 검색 시작: {query}")

            # 제한 시간 적용
            web_results = await asyncio.wait_for(
                self._perform_web_search(query, sleep_duration=sleep_duration),
                timeout=self.timeout
            )

            search_time = time.time() - start_time
            logger.debug(f"[{session_id}] 웹 검색 완료: {search_time:.4f}초 소요, {len(web_results)}개 결과")

            # 결과 캐싱
            self.cache_manager.set(cache_key, web_results)

            return web_results

        except asyncio.TimeoutError:
            logger.warning(f"[{session_id}] 웹 검색 시간 초과(> {self.timeout}초): {query}")
            return []
        except Exception as e:
            logger.error(f"[{session_id}] 웹 검색 중 오류: {e}")
            return []

    async def _perform_web_search(self, query: str, sleep_duration: float = 1.0) -> List[Document]:
        """
        실제 웹 검색을 수행합니다.

        Args:
            query: 검색 쿼리

        Returns:
            List[Document]: 검색 결과 문서 목록
        """
        # DuckDuckGo 검색이 비동기인지 확인
        is_async = asyncio.iscoroutinefunction(self.search_engine.websearch_duckduckgo_async)

        if is_async:
            try:
                # DuckDuckGo 디버그 로그 일시적으로 비활성화
                original_level = logging.getLogger().level
                logging.getLogger().setLevel(logging.WARNING)

                try:
                    return await self.search_engine.websearch_duckduckgo_async(query)
                finally:
                    # 원래 로깅 레벨 복원
                    logging.getLogger().setLevel(original_level)
            except Exception as e:
                logger.error(f"비동기 웹 검색 중 오류: {e}")
                return []
        else:
            # 동기 함수를 별도 스레드에서 실행
            return await asyncio.to_thread(self.search_engine.websearch_duckduckgo, query, sleep_duration)
