"""
모델 유틸리티 모듈

LLM 모델 관련 유틸리티 함수와 공통 기능을 제공합니다.
모델 식별, 키 생성, 체인 관리 등의 기능을 중앙화하여 코드 중복을 방지합니다.
"""

import hashlib
import logging
import time
from typing import Any, Dict, Optional, Callable

from langchain.chains.combine_documents import create_stuff_documents_chain
from src.services.utils.cache_service import CacheService

# 로거 설정
logger = logging.getLogger(__name__)


class ModelUtils:
    """
    LLM 모델 관련 유틸리티 클래스

    모델 식별, 체인 관리, 키 생성 등 모델 작업에 필요한 공통 기능을 제공합니다.
    이 클래스는 주로 정적 메서드로 구성되어 있어 인스턴스화 없이 사용할 수 있습니다.
    """

    @staticmethod
    def create_model_key(model: Any) -> str:
        """
        모델 객체로부터 캐시 키로 사용할 수 있는 문자열을 생성합니다.

        Args:
            model: LLM 모델 객체(OllamaLLM 등)

        Returns:
            str: 모델 식별 문자열
        """
        # 모델 타입을 기준으로 키 생성
        model_type = type(model).__name__

        # 만약 OllamaLLM이라면 모델 이름과 URL 추가
        if model_type == "OllamaLLM":
            model_name = getattr(model, "model", "unknown")
            base_url = getattr(model, "base_url", "local")
            temperature = getattr(model, "temperature", 0.7)
            return f"{model_type}:{model_name}:{base_url}:{temperature}"

        # 다른 모델 타입은 클래스 이름만 사용
        return model_type

    @staticmethod
    def create_prompt_key(prompt_template: Any) -> str:
        """
        프롬프트 템플릿으로부터 캐시 키로 사용할 수 있는 문자열을 생성합니다.

        Args:
            prompt_template: 프롬프트 템플릿 객체

        Returns:
            str: 프롬프트 식별 문자열
        """
        # 템플릿 문자열을 해시하여 키 생성
        if hasattr(prompt_template, "template"):
            template_str = str(prompt_template.template)
        elif hasattr(prompt_template, "messages"):
            template_str = str(prompt_template.messages)
        else:
            template_str = str(prompt_template)

        # 해시 생성
        return hashlib.md5(template_str.encode('utf-8')).hexdigest()

    @classmethod
    def get_or_create_chain(cls, settings_key: str, model: Any, prompt_template: Any) -> Any:
        """
        주어진 설정, 모델, 프롬프트 템플릿에 대한 체인을 가져오거나 생성합니다.
        OllamaLLM과 같은 해시 불가능한 객체를 안전하게 처리합니다.

        Args:
            settings_key: 설정 키
            model: LLM 모델 객체
            prompt_template: 프롬프트 템플릿

        Returns:
            Any: 체인 인스턴스(캐시된 것 또는 새로 생성된 것)
        """
        # 모델과 프롬프트에 대한 안정적인 키 생성
        model_key = cls.create_model_key(model)
        prompt_key = cls.create_prompt_key(prompt_template)

        # 결합된 캐시 키 생성
        combined_key = f"{settings_key}:{model_key}:{prompt_key}"

        def create_new_chain():
            try:
                new_chain = create_stuff_documents_chain(model, prompt_template)
                # 체인이 필요한 'invoke' 메서드를 가지고 있는지 확인
                if hasattr(new_chain, "invoke"):
                    return new_chain
                return None
            except Exception as e:
                logger.error(f"체인 생성 중 오류 발생: {str(e)}", exc_info=True)
                return None

        # 캐시에서 가져오거나 생성
        return CacheService.get_or_create("chain", combined_key, create_new_chain)

    @staticmethod
    def is_gemma_model(settings: Any) -> bool:
        """
        현재 모델이 Gemma 모델인지 확인합니다.

        Args:
            settings: 설정 객체

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

    @staticmethod
    def process_vllm_chunk(chunk: Any) -> Dict[str, Any]:
        """
        vLLM 응답 청크를 표준 형식으로 처리합니다.

        Args:
            chunk: 원시 vLLM 응답 청크

        Returns:
            Dict: 처리된 청크
        """
        # 오류 확인
        if 'error' in chunk:
            return {
                'error': True,
                'message': chunk.get('message', '알 수 없는 오류'),
                'finished': True
            }

        # 종료 마커 확인
        if chunk == '[DONE]':
            return {
                'new_text': '',
                'finished': True
            }

        # vLLM의 다양한 응답 형식 처리
        if isinstance(chunk, dict):
            # 텍스트 청크 (일반 스트리밍)
            if 'new_text' in chunk:
                return {
                    'new_text': chunk['new_text'],
                    'finished': chunk.get('finished', False)
                }
            # 완료 신호
            elif 'finished' in chunk and chunk['finished']:
                return {
                    'new_text': '',
                    'finished': True
                }
            # 전체 텍스트 응답 (비스트리밍 형식)
            elif 'generated_text' in chunk:
                return {
                    'new_text': chunk['generated_text'],
                    'finished': True
                }
            # OpenAI 호환 형식
            elif 'delta' in chunk:
                return {
                    'new_text': chunk['delta'].get('content', ''),
                    'finished': chunk.get('finished', False)
                }
            # 알 수 없는 형식
            else:
                return chunk

        # 문자열 응답 (드문 경우)
        elif isinstance(chunk, str):
            return {
                'new_text': chunk,
                'finished': False
            }

        # 기타 타입 처리
        return {
            'new_text': str(chunk),
            'finished': False
        }

    @staticmethod
    def validate_settings(settings: Any) -> None:
        """
        LLM 설정의 유효성을 검사합니다.

        Args:
            settings: 설정 객체

        Raises:
            ValueError: 필수 설정 매개변수가 누락된 경우
        """
        backend = settings.llm.llm_backend.lower()
        if backend == 'ollama':
            if not settings.ollama.model_name:
                raise ValueError("Ollama 모델 이름이 설정되지 않았습니다")
            if settings.ollama.access_type.lower() == 'url' and not settings.ollama.ollama_url:
                raise ValueError("URL 접근 방식에 Ollama URL이 설정되지 않았습니다")
        elif backend == "vllm":
            if not settings.vllm.endpoint_url:
                raise ValueError("vLLM 엔드포인트 URL이 설정되지 않았습니다")
        else:
            raise ValueError("LLM 백엔드는 'ollama' 또는 'vllm'이어야 합니다")
