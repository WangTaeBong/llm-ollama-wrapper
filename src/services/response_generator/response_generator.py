# src/services/response_generator.py
"""
응답 생성기 모듈 레거시 래퍼

이전 버전 호환성을 위한 ResponseGenerator 구현을 제공합니다.
이 모듈은 리팩토링된 구조로 자연스럽게 전환할 수 있도록 설계되었습니다.
"""

import logging
from typing import Dict, List, Tuple

from langchain_core.documents import Document

# 리팩토링된 모듈에서 실제 구현 임포트
from src.services.response_generator.core import ResponseGenerator as NewResponseGenerator

# 모듈 로거 설정
logger = logging.getLogger(__name__)


class ResponseGenerator(NewResponseGenerator):
    """
    레거시 ResponseGenerator 래퍼 클래스

    리팩토링된 ResponseGenerator와 이전 API의 호환성을 유지합니다.
    """

    def __init__(self, settings, llm_prompt_json_dict):
        """
        레거시 호환성을 위한 ResponseGenerator 초기화

        Args:
            settings: 시스템 설정 객체
            llm_prompt_json_dict: 프롬프트 정보를 포함하는 딕셔너리 객체
        """
        # 리팩토링된 클래스의 초기화 호출
        super().__init__(settings, llm_prompt_json_dict)
        logger.debug("호환성 래퍼를 통해 ResponseGenerator 초기화됨")

    # 이 클래스는 리팩토링된 클래스를 상속하므로 기본적으로 모든 메서드가 상속됩니다.
    # 필요한 경우 호환성을 위한 추가 래퍼 메서드를 여기에 추가할 수 있습니다.

    def get_translation_language_word(self, lang: str) -> Tuple[str, str, str]:
        """
        신규 구현과 호환되는 get_translation_language_word 래퍼.
        기존 호출 측에서는 이 메서드를 통해 동일한, 이전과 호환되는 방식으로
        언어 관련 데이터를 얻을 수 있습니다.

        Args:
            lang (str): 언어 코드(ko, en, jp, cn)

        Returns:
            Tuple[str, str, str]: (영어 이름, 현지 이름, 참조 표기법)
        """
        return super().get_translation_language_word(lang)

    def make_answer_reference(self, query_answer: str, rag_sys_info: str,
                              reference_word: str, retriever_documents: List[Document],
                              request=None) -> str:
        """
        호환성을 위한 make_answer_reference 래퍼.

        Args:
            query_answer (str): 원본 답변 텍스트
            rag_sys_info (str): RAG 시스템 정보
            reference_word (str): 참조 섹션 표시(예: "[References]")
            retriever_documents (List[Document]): 참조할 문서 목록
            request (Optional): 요청 데이터(기본값: None)

        Returns:
            str: 참조 정보가 추가된 답변 텍스트
        """
        return super().make_answer_reference(
            query_answer,
            rag_sys_info,
            reference_word,
            retriever_documents,
            request
        )
