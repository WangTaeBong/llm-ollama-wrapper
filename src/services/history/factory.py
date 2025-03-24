"""
히스토리 핸들러 팩토리 모듈

적절한 히스토리 핸들러 인스턴스를 생성하는 팩토리 패턴을 구현합니다.
"""

import logging
from typing import Dict, Type, Any, Optional

from src.schema.chat_req import ChatRequest
from src.services.history.base import HistoryHandlerBase
from src.services.history.handlers.history_handler import BaseHistoryHandler
from src.services.history.handlers.ollama_handler import OllamaHistoryHandler
from src.services.history.handlers.vllm_handler import VLLMHistoryHandler
from src.services.history.handlers.gemma_handler import GemmaHistoryHandler
from src.common.config_loader import ConfigLoader

# 설정 로드
settings = ConfigLoader().get_settings()

# 로거 설정
logger = logging.getLogger(__name__)


class HistoryHandlerFactory:
    """
    히스토리 핸들러 팩토리 클래스

    모델 유형에 따라 적절한 히스토리 핸들러 인스턴스를 생성합니다.
    """

    # 핸들러 레지스트리
    _registry: Dict[str, Type[HistoryHandlerBase]] = {
        "ollama": OllamaHistoryHandler,
        "vllm": VLLMHistoryHandler,
        "gemma": GemmaHistoryHandler,
    }

    # 인스턴스 캐시
    _instances: Dict[str, HistoryHandlerBase] = {}

    @classmethod
    def create_handler(
        cls,
        llm_model: Any,
        request: ChatRequest,
        max_history_turns: int = 10,
        force_new: bool = False
    ) -> HistoryHandlerBase:
        """
        히스토리 핸들러 인스턴스를 생성하거나 캐시에서 가져옵니다.

        Args:
            llm_model: LLM 모델 인스턴스
            request: 채팅 요청 객체
            max_history_turns: 최대 히스토리 턴 수
            force_new: 새 인스턴스를 강제로 생성할지 여부

        Returns:
            HistoryHandlerBase: 생성된 히스토리 핸들러 인스턴스
        """
        # 모델 유형 결정 (Gemma, vLLM, Ollama)
        model_type = cls._determine_model_type(request)
        cache_key = f"{model_type}:{request.meta.rag_sys_info}:{request.meta.session_id}"

        # 캐시에서 인스턴스 확인 (force_new가 False인 경우에만)
        if not force_new and cache_key in cls._instances:
            logger.debug(f"[{request.meta.session_id}] 캐시된 {model_type} 히스토리 핸들러 사용")
            return cls._instances[cache_key]

        # 핸들러 클래스 가져오기
        handler_class = cls._registry.get(model_type)
        if not handler_class:
            logger.warning(f"[{request.meta.session_id}] 알 수 없는 모델 유형: {model_type}, 기본 핸들러 사용")
            handler_class = BaseHistoryHandler

        # 새 인스턴스 생성
        logger.debug(f"[{request.meta.session_id}] 새 {model_type} 히스토리 핸들러 생성")
        handler = handler_class(
            llm_model=llm_model,
            request=request,
            max_history_turns=max_history_turns
        )

        # 캐시에 저장
        cls._instances[cache_key] = handler
        return handler

    @classmethod
    def _determine_model_type(cls, request: ChatRequest) -> str:
        """
        요청 데이터와 설정에서 모델 유형을 결정합니다.

        Args:
            request: 채팅 요청 객체

        Returns:
            str: 모델 유형 ("gemma", "vllm", "ollama")
        """
        # LLM 백엔드 확인
        backend = getattr(settings.llm, 'llm_backend', '').lower()

        # Ollama 백엔드 확인
        if backend == 'ollama':
            return "ollama"

        # vLLM 백엔드인 경우 Gemma 모델 확인
        elif backend == 'vllm':
            if cls._is_gemma_model():
                logger.debug(f"[{request.meta.session_id}] Gemma 모델 감지됨")
                return "gemma"
            else:
                return "vllm"

        # 기본값 반환
        logger.warning(f"[{request.meta.session_id}] 알 수 없는 백엔드: {backend}, 기본 핸들러 사용")
        return "vllm"

    @classmethod
    def _is_gemma_model(cls) -> bool:
        """
        현재 모델이 Gemma 모델인지 확인합니다.

        Returns:
            bool: Gemma 모델이면 True, 아니면 False
        """
        # LLM 백엔드 확인
        backend = getattr(settings.llm, 'llm_backend', '').lower()

        # OLLAMA 백엔드인 경우
        if backend == 'ollama':
            if hasattr(settings.ollama, 'model_name'):
                model_name = settings.ollama.model_name.lower()
                return 'gemma' in model_name

        # VLLM 백엔드인 경우
        elif backend == 'vllm':
            if hasattr(settings.llm, 'model_type'):
                model_type = settings.llm.model_type.lower() if hasattr(settings.llm.model_type, 'lower') else str(
                    settings.llm.model_type).lower()
                return model_type == 'gemma'

        # 기본적으로 False 반환
        return False

    @classmethod
    def register_handler(cls, name: str, handler_class: Type[HistoryHandlerBase]) -> None:
        """
        새로운 핸들러 클래스를 등록합니다.

        Args:
            name: 핸들러 이름
            handler_class: 핸들러 클래스
        """
        cls._registry[name.lower()] = handler_class
        logger.info(f"새 히스토리 핸들러 등록됨: {name}")

    @classmethod
    def clear_cache(cls) -> None:
        """
        핸들러 캐시를 비웁니다.
        설정이 변경되었거나 메모리를 확보해야 할 때 유용합니다.
        """
        cls._instances.clear()
        logger.debug("히스토리 핸들러 캐시가 비워졌습니다.")

    @classmethod
    def get_handler_class(cls, model_type: str) -> Optional[Type[HistoryHandlerBase]]:
        """
        지정된 모델 유형에 대한 핸들러 클래스를 반환합니다.

        Args:
            model_type: 모델 유형

        Returns:
            Optional[Type[HistoryHandlerBase]]: 핸들러 클래스 또는 None
        """
        return cls._registry.get(model_type.lower())
