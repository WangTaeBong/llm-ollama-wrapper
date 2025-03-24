"""
커스텀 검색기 모듈

다양한 소스에서 문서를 검색하는 기능을 통합적으로 제공합니다.
"""

import asyncio
import logging
import time
from collections import defaultdict
from typing import List, Dict, Any, Optional, Set

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field, field_validator

from src.common.config_loader import ConfigLoader
from src.schema.retriever_req import RetrieverRequest
from src.services.retrieval.base import DocumentStoreBase, RetrieverBase
from src.services.retrieval.cache.cache_manager import CacheManager
from src.services.retrieval.document.document_store import DocumentStore
from src.services.retrieval.providers.factory import ProviderFactory
from src.services.retrieval.utils.async_helpers import run_with_semaphore

# 로거 설정
logger = logging.getLogger(__name__)

# Load settings
settings = ConfigLoader().get_settings()


class CustomRetriever(BaseRetriever, BaseModel, RetrieverBase):
    """
    커스텀 검색기 클래스

    여러 데이터 소스에서 문서를 검색하고, LangChain의 BaseRetriever와 통합됩니다.
    동기 및 비동기 문서 검색을 모두 지원하며, 캐싱과 성능 최적화 기능을 제공합니다.
    """

    request_data: RetrieverRequest = Field(..., description="검색 요청 데이터")
    url: str = Field(default="", description="외부 API 엔드포인트 URL")
    headers: Dict[str, str] = Field(
        default={"content-type": "application/json;charset=utf-8"},
        description="HTTP 요청 헤더"
    )
    page_content_key: str = Field(default="content", description="API 응답에서 콘텐츠를 추출할 키")
    metadata_key: List[str] = Field(
        default=["doc_name", "doc_page"],
        description="API 응답에서 메타데이터를 추출할 키"
    )

    # 성능 설정
    max_concurrent_requests: int = Field(default=5, description="최대 동시 요청 수")
    cache_ttl: int = Field(default=3600, description="캐시 유효 시간(초)")

    # Pydantic 비공개 필드 (클래스 초기화 중에 초기화되지 않음)
    _document_store: DocumentStoreBase = None
    _api_provider: Any = None
    _web_provider: Any = None
    _cache_manager: CacheManager = None
    _semaphore: asyncio.Semaphore = None

    # Pydantic 필드 검증기 (V2 스타일)
    @field_validator('request_data')
    def validate_request_data(cls, v):
        """요청 데이터 검증"""
        if v is None:
            raise ValueError("request_data는 None일 수 없습니다")
        return v

    def __init__(self, **data):
        """
        CustomRetriever 초기화
        """
        super().__init__(**data)

        # 공유 캐시 관리자 초기화
        self._cache_manager = CacheManager(ttl=self.cache_ttl)

        # 의존성 초기화
        self._document_store = DocumentStore()

        # 제공자 팩토리 및 제공자 초기화
        provider_factory = ProviderFactory()
        self._api_provider = provider_factory.create_provider("api", self.request_data, self._cache_manager)
        self._web_provider = provider_factory.create_provider("web", self.request_data, self._cache_manager)

        # 동시성 제어를 위한 세마포어
        self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)

    def add_documents(self, documents: List[Document], batch_size: int = 50) -> None:
        """
        문서를 검색기의 로컬 저장소에 추가합니다.

        Args:
            documents: 저장할 Document 객체 목록
            batch_size: 성능 향상을 위해 각 배치에서 처리할 문서 수
        """
        if not documents:
            logger.warning("add_documents에 제공된 문서가 없습니다")
            return

        try:
            # 문서를 배치로 처리
            added_count = 0
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                added_count += self._document_store.add_batch(batch)

            if added_count > 0:
                logger.info(f"{added_count}개의 새 문서가 성공적으로 추가되었습니다")
        except Exception as e:
            logger.exception(f"문서 추가 중 오류: {e}")

    async def add_documents_async(self, documents: List[Document], batch_size: int = 50) -> None:
        """
        문서를 검색기의 로컬 저장소에 비동기적으로 추가합니다.

        Args:
            documents: 저장할 Document 객체 목록
            batch_size: 각 배치에서 처리할 문서 수
        """
        if not documents:
            logger.warning("add_documents_async에 제공된 문서가 없습니다")
            return

        try:
            # 문서를 배치로 처리 (비동기)
            added_count = 0
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                # 다른 비동기 작업이 진행되도록 제어 양보
                await asyncio.sleep(0)
                added_count += self._document_store.add_batch(batch)

            if added_count > 0:
                logger.info(f"{added_count}개의 새 문서가 비동기적으로 추가되었습니다")
        except Exception as e:
            logger.exception(f"문서 비동기 추가 중 오류: {e}")

    def get_all_documents(self) -> List[Document]:
        """
        저장된 모든 문서를 검색합니다.

        Returns:
            저장된 모든 Document 객체 목록
        """
        all_docs = self._document_store.get_all()
        if all_docs:
            logger.debug(f"총 {len(all_docs)}개 문서 검색됨")
        return all_docs

    def _get_relevant_documents(
            self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        문서를 동기적으로 검색합니다.

        Args:
            query: 문서 검색을 위한 쿼리 문자열
            run_manager: 선택적 콜백 관리자

        Returns:
            검색된 Document 객체 목록
        """
        session_id = self.request_data.meta.session_id
        logger.debug(f"[{session_id}] 쿼리 검색 문서: {query}")

        try:
            # 안전한 비동기 실행을 위해 asyncio.run() 사용
            return asyncio.run(self._fetch_and_process_documents(query))
        except RuntimeError as e:
            logger.error(f"[{session_id}] 이벤트 루프 오류: {e}")
            # 이벤트 루프가 이미 실행 중일 때 대체 방법
            try:
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(self._fetch_and_process_documents(query))
            except Exception as inner_e:
                logger.exception(f"[{session_id}] 대체 이벤트 루프 오류: {inner_e}")
                return []
        except Exception as e:
            logger.exception(f"[{session_id}] 동기 문서 검색 중 오류: {e}")
            return []

    async def ainvoke(
            self,
            input_string: str,
            config: Optional[Dict[str, Any]] = None,
            **kwargs
        ) -> List[Document]:
        """
        문서를 비동기적으로 검색합니다.

        LangChain의 최신 Runnable 인터페이스와 호환되는 ainvoke 메서드입니다.

        Args:
            input_string: 문서 검색을 위한 쿼리 문자열
            config: 선택적 구성 매개변수
            **kwargs: 추가 매개변수

        Returns:
            List[Document]: 검색된 Document 객체 목록
        """
        # 이전 버전 호환성을 위한 추가 인자 처리
        run_manager = kwargs.pop('run_manager', None)

        session_id = self.request_data.meta.session_id
        logger.debug(f"[{session_id}] 비동기 쿼리 검색: {input_string}")

        return await self._fetch_and_process_documents(input_string)

    async def _aget_relevant_documents(
            self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        문서를 비동기적으로 검색합니다 (LangChain 인터페이스 구현).

        Args:
            query: 문서 검색을 위한 쿼리 문자열
            run_manager: 선택적 콜백 관리자

        Returns:
            List[Document]: 검색된 Document 객체 목록
        """
        return await self.ainvoke(query, config=None, run_manager=run_manager)

    async def _fetch_and_process_documents(self, query: str) -> List[Document]:
        """
        API 및 웹 검색에서 문서를 가져와 처리하는 통합 메서드

        Args:
            query: 문서 검색을 위한 쿼리 문자열

        Returns:
            검색된 Document 객체 목록
        """
        session_id = self.request_data.meta.session_id

        try:
            start_time = time.time()
            logger.debug(f"[{session_id}] 문서 검색 프로세스 시작: {query}")

            # 세마포어 획득 (동시 요청 제한)
            async with self._semaphore:
                # 웹 검색 설정 확인
                web_search_enabled = getattr(settings.web_search, 'use_flag', False)
                document_add_type = getattr(settings.web_search, 'document_add_type', 1)

                # 작업 목록 초기화
                tasks = []
                task_types = []

                # web_search.use_flag가 true이고 document_add_type이 0이면 API 문서 검색을 건너뜀
                if not (web_search_enabled and document_add_type == 0):
                    # API 문서 검색 작업 추가
                    tasks.append(self._api_provider.fetch_documents(query, request_data=self.request_data))
                    task_types.append("api")
                    logger.debug(f"[{session_id}] API 문서 검색 작업 추가됨")
                else:
                    logger.debug(
                        f"[{session_id}] 웹 검색 설정에 따라 API 문서 검색을 건너뜁니다 "
                        f"(use_flag={web_search_enabled}, document_add_type={document_add_type})"
                    )

                # 웹 검색이 활성화된 경우 웹 검색 작업 추가
                if web_search_enabled:
                    tasks.append(self._web_provider.fetch_documents(
                        query, self.request_data.meta.rag_sys_info, session_id
                    ))
                    task_types.append("web")
                    logger.debug(f"[{session_id}] 웹 검색 작업 추가됨")

                # 모든 작업 동시 실행
                results = await asyncio.gather(*tasks)

                # 결과 처리
                rag_documents = []

                # 작업 유형에 따라 결과 처리
                for i, task_type in enumerate(task_types):
                    if task_type == "api":
                        api_response = results[i]
                        api_documents = api_response if isinstance(api_response, list) else []
                        rag_documents = api_documents
                    elif task_type == "web" and results[i]:
                        web_results = results[i]
                        # 웹 검색 결과 통합 방식 결정
                        if web_search_enabled:
                            if document_add_type == 1 and rag_documents:
                                # 기존 문서에 웹 검색 결과 추가
                                rag_documents.extend(web_results)
                            elif document_add_type == 0:
                                # 웹 검색 결과만 반환
                                rag_documents = web_results

                total_time = time.time() - start_time
                logger.debug(
                    f"[{session_id}] 문서 검색 완료: {total_time:.4f}초 소요, "
                    f"{len(rag_documents)}개 문서 발견"
                )

                return rag_documents

        except Exception as e:
            logger.exception(f"[{session_id}] 문서 가져오기 및 처리 중 오류: {e}")
            return []

    def clear_cache(self) -> None:
        """캐시를 지웁니다."""
        self._cache_manager.clear()

    # 하위 호환성을 위한 속성 접근자
    @property
    def stored_documents(self) -> Dict[str, List[Document]]:
        """저장된 문서 딕셔너리 반환 (하위 호환성용)"""
        result = defaultdict(list)
        for doc in self._document_store.get_all():
            doc_name = doc.metadata.get("doc_name", "default")
            result[doc_name].append(doc)
        return result
