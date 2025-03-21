"""
응답 생성 모듈
===========

LLM 응답을 생성하고 후처리하는 기능을 제공합니다.

기능:
- 프롬프트 템플릿 관리
- 참조 정보 추가
- 응답 형식화 및 최적화
"""

import hashlib
import logging
import re
from datetime import datetime
from functools import lru_cache
from time import localtime
from typing import Dict, List, Tuple

from langchain_core.documents import Document

from src.services.response.base import ResponseGeneratorBase
from src.services.response.factory import ResponseGeneratorFactory

# 로거 설정
logger = logging.getLogger(__name__)


class StandardResponseGenerator(ResponseGeneratorBase):
    """
    표준 응답 생성기 구현

    프롬프트 관리, 참조 정보 추가 등을 담당합니다.
    """

    def __init__(self, settings, llm_prompt_json_dict):
        """
        표준 응답 생성기 초기화

        Args:
            settings: 설정 객체
            llm_prompt_json_dict: 프롬프트 정보 딕셔너리
        """
        super().__init__(settings)
        self.llm_prompt_json_dict = llm_prompt_json_dict

        # 자주 사용되는 설정 값 캐싱
        self._cached_settings = {
            'source_rag_target': None,
            'none_source_rag_target': None,
            'faq_category_rag_target_list': None
        }

        # 언어 메타데이터 정의
        self._language_data = {
            "ko": ("Korean", "한국어", "[참고문헌]"),
            "en": ("English", "영어", "[References]"),
            "jp": ("Japanese", "일본어", "[参考文献]"),
            "cn": ("Chinese", "중국어", "[参考文献]"),
        }

        # 요일 이름 정의
        self._day_names = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']

        # 초기 설정 로드
        self._load_cached_settings()

    async def initialize(self) -> bool:
        """
        응답 생성기 초기화

        Returns:
            bool: 초기화 성공 여부
        """
        # 초기화 완료
        return True

    def _load_cached_settings(self) -> None:
        """
        자주 사용되는 설정 값 캐싱

        반복적인 설정 접근을 줄이기 위해 값을 미리 로드합니다.
        """
        try:
            # 소스 타입 설정 로드
            self._cached_settings['source_rag_target'] = self.settings.prompt.source_type.split(',')
            self._cached_settings['none_source_rag_target'] = self.settings.prompt.none_source_type.split(',')
            self._cached_settings['faq_category_rag_target_list'] = self.settings.prompt.faq_type.split(',')

            logger.debug("설정 캐싱 완료")
        except Exception as e:
            logger.warning(f"설정 캐싱 중 오류: {e}")
            # 오류 시 기본값 초기화
            for key in self._cached_settings:
                self._cached_settings[key] = []

    def _get_cached_setting(self, setting_name: str) -> List[str]:
        """
        캐시된 설정 값 조회

        Args:
            setting_name: 설정 이름

        Returns:
            List[str]: 설정 값 리스트 또는 빈 리스트
        """
        # 캐시에서 값 확인
        cached_value = self._cached_settings.get(setting_name)

        # 캐시된 값이 있으면 반환
        if cached_value is not None:
            return cached_value  # type: ignore

        # 캐시에 없으면 설정에서 로드
        try:
            if setting_name == 'source_rag_target':
                value = self.settings.prompt.source_type.split(',')
            elif setting_name == 'none_source_rag_target':
                value = self.settings.prompt.none_source_type.split(',')
            elif setting_name == 'faq_category_rag_target_list':
                value = self.settings.prompt.faq_type.split(',')
            else:
                return []

            # 캐시 업데이트
            self._cached_settings[setting_name] = value
            return value
        except Exception:
            # 오류 시 빈 리스트 반환
            return []

    async def is_faq_type_chatbot(self, current_rag_sys_info: str) -> bool:
        """
        현재 RAG 시스템 정보가 FAQ 타입인지 확인

        Args:
            current_rag_sys_info: 현재 RAG 시스템 정보

        Returns:
            bool: FAQ 타입이면 True, 아니면 False
        """
        faq_targets = self._get_cached_setting('faq_category_rag_target_list')
        return current_rag_sys_info in faq_targets

    async def is_voc_type_chatbot(self, current_rag_sys_info: str) -> bool:
        """
        현재 RAG 시스템 정보가 VOC 타입인지 확인

        Args:
            current_rag_sys_info: 현재 RAG 시스템 정보

        Returns:
            bool: VOC 타입이면 True, 아니면 False
        """
        # VOC 타입 검사
        voc_types = self.settings.voc.voc_type.split(',')
        return current_rag_sys_info in voc_types

    @lru_cache(maxsize=64)
    async def get_rag_qa_prompt(self, rag_sys_info: str) -> str:
        """
        RAG 시스템 정보에 따른 적절한 프롬프트 조회

        Args:
            rag_sys_info: RAG 시스템 정보

        Returns:
            str: 조회된 프롬프트 또는 빈 문자열
        """
        # 캐시된 설정 가져오기
        source_rag_target = self._get_cached_setting('source_rag_target')
        none_source_rag_target = self._get_cached_setting('none_source_rag_target')

        try:
            # 프롬프트 타입 결정
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
            return self.llm_prompt_json_dict.get_prompt_data("prompts", prompt_type, prompt_key) or ""
        except Exception as e:
            logger.error(f"프롬프트 조회 중 오류: {e}")
            return ""

    async def get_translation_language_word(self, lang: str) -> Tuple[str, str, str]:
        """
        언어 코드에 따른 언어명, 번역된 이름, 참조 표기 반환

        Args:
            lang: 언어 코드 (ko, en, jp, cn)

        Returns:
            Tuple[str, str, str]: (영어 이름, 현지 이름, 참조 표기)
        """
        # 유효하지 않은 언어 코드 처리
        if not lang or not isinstance(lang, str) or lang not in self._language_data:
            return self._language_data["ko"]  # 기본값으로 한국어 반환

        return self._language_data[lang]

    async def get_today(self) -> str:
        """
        한국어 형식의 현재 날짜와 요일 반환

        Returns:
            str: "YYYY년 MM월 DD일 요일 HH시 MM분" 형식 문자열
        """
        try:
            today = datetime.now()
            weekday = self._day_names[localtime().tm_wday]
            return f"{today.strftime('%Y년 %m월 %d일')} {weekday} {today.strftime('%H시 %M분')}입니다."
        except Exception as e:
            logger.warning(f"날짜 형식화 중 오류: {e}")
            # 간단한 형식으로 폴백
            return datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')

    async def make_answer_reference(self, query_answer: str, rag_sys_info: str,
                              reference_word: str, retriever_documents: List[Document], request=None) -> str:
        """
        응답에 참조 문서 정보 추가

        Args:
            query_answer: 원본 응답 텍스트
            rag_sys_info: RAG 시스템 정보
            reference_word: 참조 섹션 표시자 (예: "[References]")
            retriever_documents: 참조할 문서 리스트
            request: 요청 데이터 (선택 사항)

        Returns:
            str: 참조 정보가 추가된 응답 텍스트
        """
        # 참조 섹션 이미 존재하는지 확인
        reference_pattern = rf"---------\s*\n{re.escape(reference_word)}"
        existing_references_match = re.search(reference_pattern, query_answer)

        # 응답이 없거나 참조 문서가 없으면 원본 반환
        if not query_answer or not retriever_documents:
            return query_answer

        # 소스 타입 설정 확인
        source_rag_target = self._get_cached_setting('source_rag_target')
        if rag_sys_info not in source_rag_target:
            return query_answer

        try:
            # 문서 소스 정보 수집 (중복 제거 개선)
            docs_source = {}
            wiki_ko_docs_count = 0
            total_docs_count = 0

            # 콘텐츠 해시로 중복 문서 추적
            content_hashes = set()

            for doc in retriever_documents:
                total_docs_count += 1

                # 문서 이름 및 페이지 추출
                doc_name = doc.metadata.get("doc_name") or doc.metadata.get("source", "Unknown Document")
                if isinstance(doc_name, str) and "," in doc_name:
                    doc_name = doc_name.split(",")[0].strip()

                # wiki_ko 문서 건너뛰기
                if isinstance(doc_name, str) and "wiki_ko" in doc_name:
                    wiki_ko_docs_count += 1
                    continue

                # 콘텐츠 해시를 사용한 중복 감지
                content_hash = None
                if hasattr(doc, 'page_content') and doc.page_content:
                    # 콘텐츠 앞부분 100자를 사용하여 해시 생성
                    content_sample = doc.page_content[:100].strip()
                    content_hash = hashlib.md5(content_sample.encode('utf-8')).hexdigest()

                    # 이미 본 콘텐츠라면 건너뛰기
                    if content_hash in content_hashes:
                        continue
                    content_hashes.add(content_hash)

                doc_page = doc.metadata.get("doc_page", "N/A")

                # 웹 검색 결과 여부 판단
                is_web_result = False
                if isinstance(doc_page, str) and (doc_page.startswith("http://") or doc_page.startswith("https://")):
                    is_web_result = True
                elif isinstance(doc_name, str) and (doc_name.startswith("http://") or doc_name.startswith("https://")):
                    is_web_result = True
                    # doc_name이 URL이면 doc_page로 사용하고 doc_name 정리
                    if doc_page == "N/A":
                        doc_page = doc_name
                        # URL에서 더 깔끔한 이름 추출
                        try:
                            from urllib.parse import urlparse
                            parsed_url = urlparse(doc_name)
                            doc_name = parsed_url.netloc
                        except:
                            # 파싱 실패 시 기본값
                            doc_name = "Web Source"

                # 경로에서 선행 '/' 제거
                if isinstance(doc_name, str) and doc_name.startswith('/'):
                    doc_name = doc_name[1:]

                # 이름과 페이지를 결합한 고유 키 생성
                doc_key = f"{doc_name}_{doc_page}"

                # 동일한 소스가 없는 경우만 추가
                if doc_key not in docs_source:
                    docs_source[doc_key] = {
                        "name": doc_name,
                        "page": doc_page,
                        "is_web": is_web_result
                    }

            # 모든 문서가 wiki_ko이거나 유효한 문서가 없으면 원본 응답 반환
            if wiki_ko_docs_count == total_docs_count or not docs_source:
                return query_answer

            # 표시할 최대 소스 수 설정
            max_sources = min(
                getattr(self.settings.prompt, 'source_count', len(docs_source)),
                len(docs_source)
            )

            # 기존 참조 섹션이 있는 경우
            if existing_references_match:
                # 기존 참조 섹션 처리
                existing_section = query_answer[existing_references_match.start():]

                # 기존 문서 이름 추출
                existing_docs = set()
                for line in existing_section.split('\n'):
                    if line.startswith('- '):
                        doc_match = re.match(r'- ([^(]+)', line)
                        if doc_match:
                            existing_docs.add(doc_match.group(1).strip())

                # 새 항목 준비 (최대 소스 수 제한)
                new_entries = []
                sources_added = 0

                # 웹이 아닌 소스 먼저 처리 (일반적으로 더 신뢰성 높음)
                for doc_key, data in docs_source.items():
                    if sources_added >= max_sources:
                        break

                    doc_name = data["name"]
                    if doc_name in existing_docs:
                        continue

                    if not data["is_web"]:
                        new_entries.append(f"- {doc_name} (Page: {data['page']})")
                        existing_docs.add(doc_name)
                        sources_added += 1

                # 웹 소스도 추가 (여유 공간이 있는 경우)
                web_search_enabled = hasattr(self.settings, 'web_search') and getattr(self.settings.web_search,
                                                                                    'use_flag', False)
                if web_search_enabled:
                    for doc_key, data in docs_source.items():
                        if sources_added >= max_sources:
                            break

                        doc_name = data["name"]
                        if doc_name in existing_docs:
                            continue

                        if data["is_web"]:
                            url = data["page"]
                            if url and (url.startswith("http://") or url.startswith("https://")):
                                new_entries.append(f"- {doc_name} (<a href=\"{url}\" target=\"_blank\">Link</a>)")
                            else:
                                new_entries.append(f"- {doc_name} (Link: {url})")
                            existing_docs.add(doc_name)
                            sources_added += 1

                # 새 항목 추가 (있는 경우)
                if new_entries:
                    lines = query_answer.split('\n')
                    ref_start_idx = None
                    for i, line in enumerate(lines):
                        if reference_word in line:
                            ref_start_idx = i
                            break

                    if ref_start_idx is not None:
                        updated_lines = lines[:ref_start_idx + 1] + new_entries + lines[ref_start_idx + 1:]
                        return '\n'.join(updated_lines)

                return query_answer
            else:
                # 새 참조 섹션 생성
                # 문서 유형별 분류
                rag_results = []
                web_results = []

                for doc_key, data in docs_source.items():
                    if data["is_web"]:
                        web_results.append((data["name"], data["page"]))
                    else:
                        rag_results.append((data["name"], data["page"]))

                # 참조 섹션 생성
                reference_section = f"\n\n---------\n{reference_word}"
                sources_added = 0

                # RAG 결과 먼저 추가 (max_sources 제한)
                for i, (doc_name, doc_page) in enumerate(rag_results):
                    if sources_added >= max_sources:
                        break
                    reference_section += f"\n- {doc_name} (Page: {doc_page})"
                    sources_added += 1

                # 웹 결과 하이퍼링크 추가 (max_sources 유지)
                web_search_enabled = hasattr(self.settings, 'web_search') and getattr(self.settings.web_search,
                                                                                    'use_flag', False)
                if web_search_enabled:
                    for i, (doc_name, url) in enumerate(web_results):
                        if sources_added >= max_sources:
                            break
                        if url and (url.startswith("http://") or url.startswith("https://")):
                            reference_section += f"\n- {doc_name} (<a href=\"{url}\" target=\"_blank\">Link</a>)"
                        else:
                            reference_section += f"\n- {doc_name} (Link: {url})"
                        sources_added += 1
                else:
                    # 웹 검색 비활성화 시 링크 없이 표시
                    for i, (doc_name, url) in enumerate(web_results):
                        if sources_added >= max_sources:
                            break
                        reference_section += f"\n- {doc_name} (URL: {url})"
                        sources_added += 1

                # 원본 응답에 참조 섹션 추가
                query_answer += reference_section

                # 로깅
                if len(docs_source) > 0 and request:
                    session_id = request.meta.session_id if hasattr(request, 'meta') else 'unknown'
                    logger.debug(
                        f"[{session_id}] {sources_added}개 참조 문서가 응답에 추가됨 (총 {len(docs_source)}개 고유 소스 중)"
                    )

                return query_answer

        except Exception as e:
            logger.warning(f"참조 정보 추가 중 오류: {e}")
            return query_answer  # 오류 시 원본 응답 반환


# 클래스 등록
ResponseGeneratorFactory.register_generator("standard", StandardResponseGenerator)
