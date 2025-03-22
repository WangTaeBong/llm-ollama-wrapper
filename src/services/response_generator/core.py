# src/services/response_generator/core.py
"""
응답 생성기 핵심 모듈

채팅 응답 생성 및 서식 지정을 위한 핵심 기능을 제공합니다.
"""

import logging
from typing import Dict, List, Tuple, Optional

from langchain_core.documents import Document

from src.services.response_generator.cache.settings_cache import SettingsCache
from src.services.response_generator.formatters.reference import ReferenceFormatter
from src.services.response_generator.formatters.date import DateFormatter
from src.services.response_generator.utils.validators import DocumentValidator

# 모듈 로거 설정
logger = logging.getLogger(__name__)


class ResponseGenerator:
    """
    응답 생성 클래스

    채팅 시스템의 응답 생성, 형식 지정 및 향상 기능을 제공합니다.
    이 클래스는 다양한 형식 지정 구성 요소 및 유틸리티를 통합하여
    완전한 응답 생성 파이프라인을 제공합니다.
    """

    def __init__(self, settings, prompt_dict):
        """
        ResponseGenerator 초기화

        Args:
            settings: 시스템 설정 객체
            prompt_dict: 프롬프트 정보를 포함하는 딕셔너리 객체
        """
        self.settings = settings
        self.prompt_dict = prompt_dict

        # 캐시 관리자 초기화
        self.cache = SettingsCache(settings)

        # 포맷터 및 유틸리티 초기화
        self.reference_formatter = ReferenceFormatter(settings)
        self.date_formatter = DateFormatter()
        self.document_validator = DocumentValidator()

        # 언어별 메타데이터 초기화
        self._language_data = {
            "ko": ("Korean", "한국어", "[참고문헌]"),
            "en": ("English", "영어", "[References]"),
            "jp": ("Japanese", "일본어", "[参考文献]"),
            "cn": ("Chinese", "중국어", "[参考文献]"),
        }

        # 설정 캐싱
        self._load_cached_settings()

    def _load_cached_settings(self) -> None:
        """
        자주 사용되는 설정 값을 캐싱합니다.
        """
        try:
            self.cache.load_settings([
                'source_rag_target',
                'none_source_rag_target',
                'faq_category_rag_target_list'
            ])
            logger.debug("설정 캐싱 완료")
        except Exception as e:
            logger.error(f"설정 캐싱 중 오류 발생: {e}")

    def is_faq_type_chatbot(self, current_rag_sys_info: str) -> bool:
        """
        현재 RAG 시스템 정보가 FAQ 유형인지 확인합니다.

        Args:
            current_rag_sys_info (str): 현재 RAG 시스템 정보

        Returns:
            bool: FAQ 유형이면 True, 아니면 False
        """
        faq_targets = self.cache.get_setting('faq_category_rag_target_list', [])
        return current_rag_sys_info in faq_targets

    def is_voc_type_chatbot(self, current_rag_sys_info: str) -> bool:
        """
        현재 RAG 시스템 정보가 VOC 유형인지 확인합니다.

        Args:
            current_rag_sys_info (str): 현재 RAG 시스템 정보

        Returns:
            bool: VOC 유형이면 True, 아니면 False
        """
        try:
            voc_types = self.settings.voc.voc_type.split(',')
            return current_rag_sys_info in voc_types
        except (AttributeError, ValueError):
            logger.warning("VOC 타입 설정을 찾을 수 없습니다. 기본값 False 반환")
            return False

    def get_rag_qa_prompt(self, rag_sys_info: str) -> str:
        """
        RAG 시스템 정보에 따른 적절한 프롬프트를 가져옵니다.

        Args:
            rag_sys_info (str): RAG 시스템 정보

        Returns:
            str: 검색된 프롬프트, 찾지 못한 경우 빈 문자열
        """
        try:
            # 캐시된 설정 가져오기
            source_rag_target = self.cache.get_setting('source_rag_target', [])
            none_source_rag_target = self.cache.get_setting('none_source_rag_target', [])

            # 프롬프트 유형 결정
            if rag_sys_info in source_rag_target:
                prompt_type = "with-source-prompt"
            elif rag_sys_info in none_source_rag_target:
                prompt_type = "without-source-prompt"
            else:
                # 기본 우선순위에 따라 결정
                prompt_type = "with-source-prompt" if self.settings.prompt.source_priority else "without-source-prompt"

            # 프롬프트 키 결정
            prompt_key = (rag_sys_info
                          if rag_sys_info in source_rag_target + none_source_rag_target
                          else "common-prompt")

            # 프롬프트 가져오기
            return self.prompt_dict.get_prompt_data("prompts", prompt_type, prompt_key) or ""
        except Exception as e:
            logger.error(f"프롬프트 검색 중 오류 발생: {e}")
            return ""

    def get_translation_language_word(self, lang: str) -> Tuple[str, str, str]:
        """
        언어 코드에 따른 언어 이름, 번역된 이름 및 참조 표기법을 반환합니다.

        Args:
            lang (str): 언어 코드(ko, en, jp, cn)

        Returns:
            Tuple[str, str, str]: (영어 이름, 현지 이름, 참조 표기법)
        """
        # 잘못된 언어 코드 처리
        if not lang or not isinstance(lang, str) or lang not in self._language_data:
            return self._language_data["ko"]  # 한국어를 기본값으로 반환

        return self._language_data[lang]

    def get_today(self) -> str:
        """
        한국어 형식의 현재 날짜와 요일을 반환합니다.

        Returns:
            str: "YYYY년 MM월 DD일 요일 HH시 MM분" 형식의 문자열
        """
        return self.date_formatter.get_formatted_date()

    def make_answer_reference(self, query_answer: str, rag_sys_info: str,
                              reference_word: str, retriever_documents: List[Document],
                              request=None) -> str:
        """
        중복 제거 및 소스 수 제한으로 답변에 참조 문서 정보를 추가합니다.

        Args:
            query_answer (str): 원본 답변 텍스트
            rag_sys_info (str): RAG 시스템 정보
            reference_word (str): 참조 섹션 표시(예: "[References]")
            retriever_documents (List[Document]): 참조할 문서 목록
            request (Optional): 요청 데이터(기본값: None)

        Returns:
            str: 참조 정보가 추가된 답변 텍스트
        """
        return self.reference_formatter.add_references(
            query_answer,
            rag_sys_info,
            reference_word,
            retriever_documents,
            request
        )
