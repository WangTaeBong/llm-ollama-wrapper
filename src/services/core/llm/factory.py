"""
LLM 서비스 팩토리 모듈
===================

이 모듈은 설정에 따라 적절한 LLM 서비스 구현체를 생성하는 팩토리 패턴을 구현합니다.
의존성 주입 원칙에 따라 필요한 구성 요소를 LLM 서비스에 제공합니다.

기능:
- 설정 기반 LLM 서비스 인스턴스 생성
- 서비스 구현체 관리
- 서비스 상태 모니터링
"""

import logging
from typing import Dict, Type, Optional, Any

from .base import LLMServiceBase

# 나중에 구현될 클래스들을 미리 임포트 선언
# from .ollama import OllamaLLMService
# from .vllm import VLLMLLMService

# 로거 설정
logger = logging.getLogger(__name__)


class LLMServiceFactory:
    """
    LLM 서비스 팩토리 클래스

    설정에 따라 적절한 LLM 서비스 구현체를 생성하고 관리합니다.
    """

    # 사용 가능한 LLM 서비스 매핑
    _service_registry: Dict[str, Type[LLMServiceBase]] = {}

    # 활성화된 서비스 인스턴스 (싱글톤)
    _active_service: Optional[LLMServiceBase] = None

    @classmethod
    def register_service(cls, name: str, service_class: Type[LLMServiceBase]) -> None:
        """
        LLM 서비스 구현체 등록

        Args:
            name: 서비스 이름
            service_class: LLM 서비스 클래스
        """
        if not issubclass(service_class, LLMServiceBase):
            raise TypeError(f"{service_class.__name__}는 LLMServiceBase를 상속받아야 합니다.")

        cls._service_registry[name.lower()] = service_class
        logger.debug(f"LLM 서비스 '{name}' 등록 완료")

    @classmethod
    async def create_service(cls, settings: Any, backend: Optional[str] = None) -> LLMServiceBase:
        """
        설정 기반으로 LLM 서비스 생성

        Args:
            settings: 설정 객체
            backend: 백엔드 이름 (설정에서 읽지 않고 강제 지정할 경우)

        Returns:
            LLMServiceBase: 생성된 LLM 서비스

        Raises:
            ValueError: 지원되지 않는 백엔드 요청 시
        """
        # 이미 활성화된 서비스가 있으면 재사용
        if cls._active_service is not None:
            return cls._active_service

        # 명시적 백엔드 또는 설정에서 백엔드 읽기
        backend_name = backend or getattr(settings.llm, 'llm_backend', '').lower()

        if not backend_name:
            raise ValueError("LLM 백엔드가 설정되지 않았습니다.")

        # 지원되는 백엔드인지 확인
        if backend_name not in cls._service_registry:
            supported = ", ".join(cls._service_registry.keys())
            raise ValueError(
                f"'{backend_name}'은 지원되지 않는 LLM 백엔드입니다. 지원되는 백엔드: {supported}"
            )

        # 서비스 생성 및 초기화
        service_class = cls._service_registry[backend_name]
        service = service_class(settings)

        # 비동기 초기화 수행
        is_initialized = await service.initialize()
        if not is_initialized:
            raise RuntimeError(f"{backend_name} LLM 서비스 초기화에 실패했습니다.")

        # 활성화된 서비스로 설정
        cls._active_service = service
        logger.info(f"{backend_name} LLM 서비스 생성 및 초기화 완료")

        return service

    @classmethod
    def get_active_service(cls) -> Optional[LLMServiceBase]:
        """
        현재 활성화된 LLM 서비스 반환

        Returns:
            Optional[LLMServiceBase]: 활성화된 서비스 또는 None
        """
        return cls._active_service

    @classmethod
    def reset(cls) -> None:
        """
        활성화된 서비스 리셋 (주로 테스트용)
        """
        cls._active_service = None
        logger.debug("LLM 서비스 팩토리 리셋 완료")
