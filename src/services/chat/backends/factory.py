"""
백엔드 팩토리 모듈

요청 및 설정에 기반하여 적절한 LLM 백엔드 인스턴스를 생성합니다.
"""

import logging
from typing import Dict, Type, Any

from src.common.config_loader import ConfigLoader
from src.schema.chat_req import ChatRequest
from src.services.chat.backends.base import LLMBackend

# 설정 로드
settings = ConfigLoader().get_settings()

# 로거 설정
logger = logging.getLogger(__name__)


class BackendFactory:
    """
    LLM 백엔드 팩토리 클래스

    설정 및 요청에 따라 적절한 LLM 백엔드 인스턴스를 생성합니다.
    """

    # 백엔드 유형별 클래스 레지스트리
    _registry: Dict[str, Type[LLMBackend]] = {}

    # 백엔드 인스턴스 캐시
    _instances: Dict[str, LLMBackend] = {}

    @classmethod
    def register_backend(cls, backend_type: str, backend_class: Type[LLMBackend]) -> None:
        """
        백엔드 유형과 클래스를 등록합니다.

        Args:
            backend_type: 백엔드 유형 식별자
            backend_class: 백엔드 클래스
        """
        cls._registry[backend_type.lower()] = backend_class
        logger.info(f"LLM 백엔드 등록: {backend_type}")

    @classmethod
    def create_backend(cls, request: ChatRequest) -> LLMBackend:
        """
        요청에 맞는 LLM 백엔드 인스턴스를 생성합니다.

        Args:
            request: 채팅 요청

        Returns:
            LLMBackend: 생성된 백엔드 인스턴스

        Raises:
            ValueError: 알 수 없는 백엔드 유형
        """
        # 설정에서 백엔드 유형 가져오기
        backend_type = settings.llm.llm_backend.lower()

        # 백엔드 클래스 가져오기
        if backend_type not in cls._registry:
            # 지연 초기화: 필요시 클래스 임포트
            if backend_type == "ollama":
                # Ollama 백엔드 등록
                from src.services.chat.backends.ollama import OllamaBackend
                cls.register_backend("ollama", OllamaBackend)
            elif backend_type == "vllm":
                # vLLM 백엔드 등록
                from src.services.chat.backends.vllm import VLLMBackend
                cls.register_backend("vllm", VLLMBackend)
            else:
                raise ValueError(f"알 수 없는 LLM 백엔드 유형: {backend_type}")

        # 세션별 백엔드 캐시 키 생성
        cache_key = f"{backend_type}:{request.meta.session_id}"

        # 기존 인스턴스가 있으면 반환
        if cache_key in cls._instances:
            return cls._instances[cache_key]

        # 새 인스턴스 생성
        backend_class = cls._registry[backend_type]
        instance = backend_class(request)

        # 인스턴스 캐싱
        cls._instances[cache_key] = instance
        logger.debug(f"[{request.meta.session_id}] 새 {backend_type} 백엔드 인스턴스 생성")

        return instance

    @classmethod
    def clear_cache(cls) -> None:
        """백엔드 인스턴스 캐시를 비웁니다."""
        cls._instances.clear()
        logger.info("백엔드 인스턴스 캐시 초기화 완료")

    @classmethod
    def get_backend_class(cls, backend_type: str) -> Type[LLMBackend]:
        """
        백엔드 유형에 해당하는 클래스를 반환합니다.

        Args:
            backend_type: 백엔드 유형 식별자

        Returns:
            Type[LLMBackend]: 백엔드 클래스

        Raises:
            ValueError: 알 수 없는 백엔드 유형
        """
        backend_type = backend_type.lower()

        # 지연 초기화: 필요시 클래스 임포트
        if backend_type not in cls._registry:
            if backend_type == "ollama":
                from src.services.chat.backends.ollama import OllamaBackend
                cls.register_backend("ollama", OllamaBackend)
            elif backend_type == "vllm":
                from src.services.chat.backends.vllm import VLLMBackend
                cls.register_backend("vllm", VLLMBackend)
            else:
                raise ValueError(f"알 수 없는 LLM 백엔드 유형: {backend_type}")

        return cls._registry[backend_type]
