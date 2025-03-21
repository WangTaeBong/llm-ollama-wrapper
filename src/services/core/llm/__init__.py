"""
LLM 서비스 모듈
=============

이 모듈은 다양한 백엔드(Ollama, vLLM)를 통해 대규모 언어 모델과 상호작용하기 위한
통합 인터페이스를 제공합니다. 설정에 기반하여 적절한 서비스 구현체를 인스턴스화하는
팩토리 패턴을 따릅니다.

사용 예시:
    from src.services.core.llm import LLMServiceFactory

    # 설정에 기반한 서비스 생성
    llm_service = await LLMServiceFactory.create_service(settings)

    # 서비스 사용
    response = await llm_service.ask(query, documents, language)
"""

from .base import LLMServiceBase
from .factory import LLMServiceFactory

# 팩토리에 등록되도록 구현체 임포트
from .ollama import OllamaLLMService
from .vllm import VLLMLLMService

__all__ = [
    'LLMServiceBase',
    'LLMServiceFactory',
    'OllamaLLMService',
    'VLLMLLMService',
]