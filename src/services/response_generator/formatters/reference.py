# src/services/response_generator/formatters/reference.py
"""
참조 포맷 모듈

응답에 문서 참조를 추가하는 기능을 제공합니다.
"""

import hashlib
import logging
import re
from typing import List, Dict, Any, Set, Optional, Tuple

from langchain_core.documents import Document

# 모듈 로거 설정
logger = logging.getLogger(__name__)


class ReferenceFormatter:
    """
    참조 포맷 클래스

    응답 텍스트에 문서 참조를 추가하고 서식을 지정하는 기능을 제공합니다.
    """

    def __init__(self, settings):
        """
        ReferenceFormatter 초기화

        Args:
            settings: 시스템 설정 객체
        """
        self.settings = settings
        self._source_rag_targets = self._get_source_rag_targets()
        self._url_pattern = re.compile(
            r'https?://(?:www\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b[-a-zA-Z0-9@:%_+.~#?&/=]*')
        self._doc_name_pattern = re.compile(r'- ([^(]+)')

    def _get_source_rag_targets(self) -> List[str]:
        """
        소스 RAG 대상 목록을 가져옵니다.

        Returns:
            List[str]: 소스 RAG 대상 목록
        """
        try:
            return self.settings.prompt.source_type.split(',')
        except (AttributeError, ValueError):
            logger.warning("소스 RAG 대상 설정을 찾을 수 없습니다. 빈 목록 반환")
            return []

    def _is_source_type(self, rag_sys_info: str) -> bool:
        """
        RAG 시스템 정보가 소스 유형인지 확인합니다.

        Args:
            rag_sys_info: RAG 시스템 정보

        Returns:
            bool: 소스 유형이면 True, 아니면 False
        """
        return rag_sys_info in self._source_rag_targets

    def add_references(self, query_answer: str, rag_sys_info: str,
                       reference_word: str, retriever_documents: List[Document],
                       request=None) -> str:
        """
        응답에 참조 문서 정보를 추가합니다.

        Args:
            query_answer: 원본 응답 텍스트
            rag_sys_info: RAG 시스템 정보
            reference_word: 참조 섹션 지시자(예: "[References]")
            retriever_documents: 참조할 문서 목록
            request: 요청 데이터(선택 사항)

        Returns:
            str: 참조 정보가 추가된 응답 텍스트
        """
        # 참조 섹션이 이미 존재하는지 확인
        reference_pattern = rf"---------\s*\n{re.escape(reference_word)}"
        existing_references_match = re.search(reference_pattern, query_answer)

        # 응답이나 참조 문서가 없으면 원본 반환
        if not query_answer or not retriever_documents:
            return query_answer

        # 소스 유형 설정 확인
        if not self._is_source_type(rag_sys_info):
            return query_answer

        try:
            # 중복 제거된 문서 소스 정보 수집
            docs_data = self._collect_document_sources(retriever_documents)

            # 문서가 없으면 원본 반환
            if not docs_data:
                return query_answer

            # 최대 소스 수 가져오기
            max_sources = min(
                getattr(self.settings.prompt, 'source_count', len(docs_data)),
                len(docs_data)
            )

            # 기존 참조 처리 또는 새 참조 섹션 생성
            if existing_references_match:
                return self._update_existing_references(
                    query_answer,
                    existing_references_match,
                    docs_data,
                    max_sources
                )
            else:
                return self._create_new_references(
                    query_answer,
                    reference_word,
                    docs_data,
                    max_sources
                )

        except Exception as e:
            logger.warning(f"참조 정보 추가 중 오류 발생: {e}")
            return query_answer  # 오류 발생 시 원본 반환

    def _collect_document_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        문서 소스 정보를 수집하고 중복을 제거합니다.

        Args:
            documents: 문서 목록

        Returns:
            List[Dict[str, Any]]: 중복 제거된 소스 정보
        """
        # 문서 소스 정보 수집
        docs_data = []
        content_hashes: Set[str] = set()
        wiki_ko_docs_count = 0
        total_docs_count = 0

        for doc in documents:
            total_docs_count += 1

            # 문서 이름 및 페이지 추출
            doc_name = doc.metadata.get("doc_name") or doc.metadata.get("source", "Unknown Document")
            if isinstance(doc_name, str) and "," in doc_name:
                doc_name = doc_name.split(",")[0].strip()

            # wiki_ko 문서 건너뛰기
            if isinstance(doc_name, str) and "wiki_ko" in doc_name:
                wiki_ko_docs_count += 1
                continue

            # 중복 감지를 위한 문서 내용 해시 생성
            if hasattr(doc, 'page_content') and doc.page_content:
                # 해시를 위해 콘텐츠의 처음 100자 사용
                content_sample = doc.page_content[:100].strip()
                content_hash = hashlib.md5(content_sample.encode('utf-8')).hexdigest()

                # 이미 처리한 콘텐츠면 건너뛰기
                if content_hash in content_hashes:
                    continue
                content_hashes.add(content_hash)

            doc_page = doc.metadata.get("doc_page", "N/A")

            # 웹 검색 결과인지 확인
            is_web_result = self._is_web_result(doc_name, doc_page)

            # URL인 경우 doc_page와 doc_name 보정
            if is_web_result and isinstance(doc_name, str) and self._is_url(doc_name) and doc_page == "N/A":
                doc_page = doc_name
                doc_name = self._extract_domain(doc_name)

            # 경로에서 선행 '/' 제거
            if isinstance(doc_name, str) and doc_name.startswith('/'):
                doc_name = doc_name[1:]

            # 소스 정보 저장
            docs_data.append({
                "name": doc_name,
                "page": doc_page,
                "is_web": is_web_result
            })

        # 모든 문서가 wiki_ko이거나 유효한 문서가 없으면 빈 목록 반환
        if wiki_ko_docs_count == total_docs_count or not docs_data:
            return []

        return docs_data

    def _is_web_result(self, doc_name: Optional[str], doc_page: Optional[str]) -> bool:
        """
        문서가 웹 검색 결과인지 확인합니다.

        Args:
            doc_name: 문서 이름 (None일 수 있음)
            doc_page: 문서 페이지 정보 (None일 수 있음)

        Returns:
            bool: 웹 검색 결과이면 True, 아니면 False
        """
        if isinstance(doc_page, str) and self._is_url(doc_page):
            return True
        if isinstance(doc_name, str) and self._is_url(doc_name):
            return True
        return False

    def _is_url(self, text: Optional[str]) -> bool:
        """
        텍스트가 URL인지 확인합니다.

        Args:
            text: 확인할 텍스트 (None일 수 있음)

        Returns:
            bool: URL이면 True, 아니면 False
        """
        return bool(text and self._url_pattern.match(text))

    def _extract_domain(self, url: str) -> str:
        """
        URL에서 도메인 이름을 추출합니다.

        Args:
            url: 도메인을 추출할 URL

        Returns:
            str: 추출된 도메인 또는 기본값
        """
        if not url or not isinstance(url, str):
            return "Web Source"

        try:
            from urllib.parse import urlparse
            parsed_url = urlparse(url)
            domain = parsed_url.netloc

            # 도메인이 비어있는 경우 기본값 반환
            if not domain:
                return "Web Source"

            return domain
        except ValueError as e:
            # URL 형식 오류 처리
            logger.warning(f"잘못된 URL 형식: {url}, 오류: {e}")
            return "Web Source"
        except ImportError as e:
            # urlparse 가져오기 실패 처리
            logger.error(f"urlparse 가져오기 실패: {e}")
            return "Web Source"
        except Exception as e:
            # 기타 예외는 여전히 포괄적으로 처리하되 로깅
            logger.error(f"URL 처리 중 예상치 못한 오류: {e}, URL: {url}")
            return "Web Source"

    def _update_existing_references(self, query_answer: str, match: re.Match,
                                    docs_data: List[Dict[str, Any]], max_sources: int) -> str:
        """
        기존 참조 섹션을 업데이트합니다.

        Args:
            query_answer: 원본 응답 텍스트
            match: 기존 참조 섹션의 정규식 매치
            docs_data: 문서 소스 데이터
            max_sources: 최대 소스 수

        Returns:
            str: 업데이트된 참조가 있는 응답 텍스트
        """
        existing_section = query_answer[match.start():]

        # 기존 문서 이름 추출
        existing_docs = set()
        for line in existing_section.split('\n'):
            if line.startswith('- '):
                doc_match = self._doc_name_pattern.match(line)
                if doc_match:
                    existing_docs.add(doc_match.group(1).strip())  # type: ignore[arg-type]

        # 새 항목 준비, max_sources 제한 적용
        new_entries = []
        sources_added = 0

        # 비웹 소스 먼저 처리(일반적으로 더 신뢰할 수 있음)
        new_entries, sources_added = self._process_non_web_sources(
            docs_data, existing_docs, sources_added, max_sources
        )

        # 웹 소스 처리(여전히 공간이 있는 경우)
        web_search_enabled = self._is_web_search_enabled()
        if web_search_enabled and sources_added < max_sources:
            web_entries, total_added = self._process_web_sources(
                docs_data, existing_docs, sources_added, max_sources
            )
            new_entries.extend(web_entries)
            sources_added = total_added

        # 새 항목이 있으면 추가
        if new_entries:
            lines = query_answer.split('\n')
            ref_start_idx = None
            for i, line in enumerate(lines):
                if "[" in line and "]" in line and "참고" in line:  # 참조 섹션 검색 일반화
                    ref_start_idx = i
                    break

            if ref_start_idx is not None:
                updated_lines = lines[:ref_start_idx + 1] + new_entries + lines[ref_start_idx + 1:]
                return '\n'.join(updated_lines)

        return query_answer

    def _process_non_web_sources(self, docs_data: List[Dict[str, Any]],
                                 existing_docs: Set[str], sources_added: int,
                                 max_sources: int) -> Tuple[List[str], int]:
        """
        비웹 소스를 처리하고 참조 항목을 생성합니다.

        Args:
            docs_data: 문서 소스 데이터
            existing_docs: 이미 존재하는 문서 이름 집합
            sources_added: 현재까지 추가된 소스 수
            max_sources: 최대 소스 수

        Returns:
            Tuple[List[str], int]: 참조 항목 목록과 총 추가된 소스 수
        """
        entries = []
        for data in docs_data:
            if sources_added >= max_sources:
                break

            doc_name = data["name"]
            if doc_name in existing_docs:
                continue

            if not data["is_web"]:
                entries.append(f"- {doc_name} (Page: {data['page']})")
                existing_docs.add(doc_name)
                sources_added += 1

        return entries, sources_added

    def _process_web_sources(self, docs_data: List[Dict[str, Any]],
                             existing_docs: Set[str], sources_added: int,
                             max_sources: int) -> Tuple[List[str], int]:
        """
        웹 소스를 처리하고 참조 항목을 생성합니다.

        Args:
            docs_data: 문서 소스 데이터
            existing_docs: 이미 존재하는 문서 이름 집합
            sources_added: 현재까지 추가된 소스 수
            max_sources: 최대 소스 수

        Returns:
            Tuple[List[str], int]: 참조 항목 목록과 총 추가된 소스 수
        """
        entries = []
        for data in docs_data:
            if sources_added >= max_sources:
                break

            doc_name = data["name"]
            if doc_name in existing_docs:
                continue

            if data["is_web"]:
                url = data["page"]
                if url and self._is_url(url):
                    entries.append(f"- {doc_name} (<a href=\"{url}\" target=\"_blank\">Link</a>)")
                else:
                    entries.append(f"- {doc_name} (Link: {url})")
                existing_docs.add(doc_name)
                sources_added += 1

        return entries, sources_added

    def _is_web_search_enabled(self) -> bool:
        """
        웹 검색 기능이 활성화되어 있는지 확인합니다.

        Returns:
            bool: 웹 검색이 활성화되어 있으면 True, 아니면 False
        """
        try:
            return hasattr(self.settings, 'web_search') and getattr(self.settings.web_search, 'use_flag', False)
        except Exception as e:
            logger.error(f"web_search 설정 부분이 잘못되어 오류: {e}")
            return False

    def _create_new_references(self, query_answer: str, reference_word: str,
                               docs_data: List[Dict[str, Any]], max_sources: int) -> str:
        """
        새 참조 섹션을 생성합니다.

        Args:
            query_answer: 원본 응답 텍스트
            reference_word: 참조 섹션 표시자
            docs_data: 문서 소스 데이터
            max_sources: 최대 소스 수

        Returns:
            str: 새 참조가 추가된 응답 텍스트
        """
        # 문서 유형별로 분리
        rag_docs = []
        web_docs = []

        for data in docs_data:
            if data["is_web"]:
                web_docs.append((data["name"], data["page"]))
            else:
                rag_docs.append((data["name"], data["page"]))

        # 참조 섹션 생성
        reference_section = f"\n\n---------\n{reference_word}"
        sources_added = 0

        # RAG 결과 먼저 추가(max_sources로 제한)
        for i, (doc_name, doc_page) in enumerate(rag_docs):
            if sources_added >= max_sources:
                break
            reference_section += f"\n- {doc_name} (Page: {doc_page})"
            sources_added += 1

        # 웹 검색 결과를 하이퍼링크로 추가(여전히 max_sources 준수)
        web_search_enabled = self._is_web_search_enabled()
        if web_search_enabled:
            for i, (doc_name, url) in enumerate(web_docs):
                if sources_added >= max_sources:
                    break
                if url and self._is_url(url):
                    reference_section += f"\n- {doc_name} (<a href=\"{url}\" target=\"_blank\">Link</a>)"
                else:
                    reference_section += f"\n- {doc_name} (Link: {url})"
                sources_added += 1
        else:
            # 웹 검색이 비활성화된 경우 링크 없이 결과 표시
            for i, (doc_name, url) in enumerate(web_docs):
                if sources_added >= max_sources:
                    break
                reference_section += f"\n- {doc_name} (URL: {url})"
                sources_added += 1

        # 원본 응답에 참조 섹션 추가
        return query_answer + reference_section
