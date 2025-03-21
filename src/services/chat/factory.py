"""
Chat 서비스 팩토리 모듈
===================

이 모듈은 설정에 따라 적절한 Chat 서비스 구현체를 생성하는 팩토리 패턴을 구현합니다.
의존성 주입 원칙에 따라 필요한 구성 요소를 Chat 서비스에 제공합니다.

기능:
- 설정 기반 Chat 서비스 인스턴스 생성
- 서비스 구성 요소 관리
- 서비스 상태 모니터링
"""

import logging
from typing import Dict, Optional, Type

from src.common.config_loader import ConfigLoader
from src.schema.chat_req import ChatRequest
from src.services.chat.service import ChatService
from src.services.core.llm import LLMServiceFactory
from src.services.retrieval import RetrievalSourceFactory
from src.common.query_check_dict import QueryCheckDict

# 로거 설정
logger = logging.getLogger(__name__)


class ChatServiceFactory:
    """
    Chat 서비스 팩토리 클래스

    요청 정보에 따라 적절한 Chat 서비스 인스턴스를 생성하고 관리합니다.
    모든 필요한 의존성을 주입합니다.
    """

    # 서비스 인스턴스 캐시 (session_id 기준)
    _service_instances: Dict[str, ChatService] = {}

    # 설정 객체 (캐싱)
    _settings = None

    # 쿼리 체크 딕셔너리 (캐싱)
    _query_check_dict = None

    @classmethod
    async def create_service(cls, request: ChatRequest) -> ChatService:
        """
        요청 정보에 따라 Chat 서비스 인스턴스 생성

        이미 생성된 인스턴스가 있으면 재사용하고,
        없으면 신규 인스턴스를 생성합니다.

        Args:
            request (ChatRequest): 채팅 요청 객체

        Returns:
            ChatService: 생성된 Chat 서비스 인스턴스
        """
        # 설정 로드 (최초 1회)
        if cls._settings is None:
            cls._settings = ConfigLoader().get_settings()

        # 쿼리 체크 딕셔너리 로드 (최초 1회)
        if cls._query_check_dict is None:
            cls._query_check_dict = QueryCheckDict(cls._settings.prompt.llm_prompt_path)

        # 세션 ID 추출
        session_id = request.meta.session_id

        # 인스턴스 캐시 키 생성
        # (세션별로 다른 인스턴스를 사용하게 함)
        cache_key = f"{request.meta.rag_sys_info}:{session_id}"

        # 캐시된 인스턴스가 있으면 재사용
        if cache_key in cls._service_instances:
            service = cls._service_instances[cache_key]
            # 요청 정보 업데이트
            service.update_request(request)
            logger.debug(f"[{session_id}] 캐시된 ChatService 인스턴스 재사용")
            return service

        # 필요한 서비스 컴포넌트 생성
        try:
            # LLM 서비스 생성
            llm_service = await LLMServiceFactory.create_service(cls._settings)

            # 검색 소스 생성
            retrieval_source = await RetrievalSourceFactory.create_service(cls._settings)

            # 채팅 서비스 인스턴스 생성
            service = ChatService(
                request=request,
                settings=cls._settings,
                llm_service=llm_service,
                retrieval_source=retrieval_source,
                query_check_dict=cls._query_check_dict
            )

            # 캐시에 인스턴스 저장
            cls._service_instances[cache_key] = service

            logger.debug(f"[{session_id}] 새 ChatService 인스턴스 생성 완료")
            return service

        except Exception as e:
            logger.error(f"[{session_id}] ChatService 인스턴스 생성 중 오류: {str(e)}")
            raise RuntimeError(f"ChatService 생성 실패: {str(e)}")

    @classmethod
    def get_service(cls, session_id: str, rag_sys_info: str) -> Optional[ChatService]:
        """
        세션 ID와 RAG 시스템 정보로 서비스 인스턴스 조회

        Args:
            session_id (str): 세션 ID
            rag_sys_info (str): RAG 시스템 정보

        Returns:
            Optional[ChatService]: 찾은 서비스 인스턴스 또는 None
        """
        cache_key = f"{rag_sys_info}:{session_id}"
        return cls._service_instances.get(cache_key)

    @classmethod
    def remove_service(cls, session_id: str, rag_sys_info: str) -> None:
        """
        서비스 인스턴스 제거

        Args:
            session_id (str): 세션 ID
            rag_sys_info (str): RAG 시스템 정보
        """
        cache_key = f"{rag_sys_info}:{session_id}"
        if cache_key in cls._service_instances:
            del cls._service_instances[cache_key]
            logger.debug(f"[{session_id}] ChatService 인스턴스 제거됨")

    @classmethod
    def cleanup_old_instances(cls, max_instances: int = 100) -> None:
        """
        오래된 서비스 인스턴스 정리

        Args:
            max_instances (int): 유지할 최대 인스턴스 수
        """
        if len(cls._service_instances) > max_instances:
            # 가장 오래된 인스턴스부터 제거
            overflow = len(cls._service_instances) - max_instances
            keys_to_remove = list(cls._service_instances.keys())[:overflow]

            for key in keys_to_remove:
                del cls._service_instances[key]

            logger.info(f"{overflow}개의 오래된 ChatService 인스턴스 정리됨")
