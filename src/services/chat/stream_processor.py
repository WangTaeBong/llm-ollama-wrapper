"""
스트리밍 응답 처리 모듈
===================

스트리밍 방식의 응답을 효율적으로 처리하는 기능을 제공합니다.

기능:
- 문자 단위 응답 처리
- URL 및 특수 패턴 감지
- 참조 정보 추가
"""

import asyncio
import logging
import re
import time
from typing import Dict, List, Any, Tuple

from langchain_core.documents import Document

# 로거 설정
logger = logging.getLogger(__name__)


class StreamResponseProcessor:
    """
    스트리밍 응답 처리기

    문자 단위로 응답을 처리하여 빠른 사용자 경험을 제공합니다.
    """

    def __init__(self, response_generator, voc_processor, request, documents):
        """
        스트리밍 응답 처리기 초기화

        Args:
            response_generator: 응답 생성기
            voc_processor: VOC 처리기
            request: 채팅 요청 객체
            documents: 검색된 문서 리스트
        """
        self.response_generator = response_generator
        self.voc_processor = voc_processor
        self.request = request
        self.documents = documents
        self.logger = logging.getLogger(__name__)

        # 전체 응답 저장
        self.full_text = ""
        self.processed_chunks = []

        # 처리 설정
        self.min_chars = 2  # 한글은 자모 조합 고려해 최소 2자 이상일 때 전송
        self.force_interval = 100  # 최대 100ms 이상 지연되지 않도록 함
        self.last_send_time = time.time()

        # URL 및 특수 패턴 감지
        self.url_pattern = re.compile(r'https?://\S+')
        self.url_buffer = ""  # URL 완성까지 임시 저장
        self.in_url = False  # URL 처리 중 상태

    async def process_partial(self, text: str) -> Tuple[str, str]:
        """
        문자 단위 텍스트 처리 - 문장 완성을 기다리지 않음

        Args:
            text: 처리할 텍스트

        Returns:
            tuple: (처리된_텍스트, 남은_버퍼)
        """
        current_time = time.time()
        force_send = (current_time - self.last_send_time) > (self.force_interval / 1000)

        # 텍스트가 없으면 처리하지 않음
        if not text:
            return "", ""

        # URL 패턴 검사 - URL은 완성될 때까지 버퍼링
        if self.in_url:
            # URL 종료 조건 확인 (공백, 줄바꿈 등)
            end_idx = -1
            for i, char in enumerate(text):
                if char.isspace():
                    end_idx = i
                    break

            if end_idx >= 0:
                # URL 완성됨
                self.url_buffer += text[:end_idx]
                processed_url = self._quick_process_urls(self.url_buffer)

                # 처리 결과와 남은 텍스트 반환
                self.in_url = False
                self.full_text += self.url_buffer + text[end_idx:end_idx + 1]
                remaining = text[end_idx + 1:]
                self.url_buffer = ""

                self.last_send_time = current_time
                return processed_url + text[end_idx:end_idx + 1], remaining
            else:
                # URL 계속 축적
                self.url_buffer += text
                self.full_text += text
                return "", ""  # URL 완성될 때까지 출력 보류

        # URL 시작 감지
        url_match = self.url_pattern.search(text)
        if url_match:
            start_idx = url_match.start()
            if start_idx > 0:
                # URL 이전 텍스트 처리
                prefix = text[:start_idx]
                self.full_text += prefix

                # URL 부분 버퍼링 시작
                self.in_url = True
                self.url_buffer = text[start_idx:]

                self.last_send_time = current_time
                return prefix, ""
            else:
                # 텍스트가 URL로 시작함
                self.in_url = True
                self.url_buffer = text
                self.full_text += text
                return "", ""

        # 일반 텍스트 처리 (URL 아님)
        # 충분한 텍스트가 있거나 강제 전송 조건 충족 시 전송
        if len(text) >= self.min_chars or force_send:
            self.full_text += text
            self.last_send_time = current_time
            return text, ""

        # 최소 길이 미달 시 버퍼 유지
        self.full_text += text
        return "", ""

    def _quick_process_urls(self, text: str) -> str:
        """
        URL을 빠르게 링크로 변환

        Args:
            text: URL을 포함한 텍스트

        Returns:
            str: 링크로 변환된 텍스트
        """
        return self.url_pattern.sub(lambda m: f'<a href="{m.group(0)}" target="_blank">{m.group(0)}</a>', text)

    async def finalize(self, remaining_text: str) -> str:
        """
        최종 처리 - 참조 및 VOC 처리 등 무거운 작업 수행

        Args:
            remaining_text: 남은 텍스트

        Returns:
            str: 최종 처리된 텍스트
        """
        session_id = self.request.meta.session_id
        self.logger.debug(f"[{session_id}] 응답 최종 처리 시작")

        # 남은 텍스트 및 URL 버퍼 처리
        final_text = remaining_text
        if self.url_buffer:
            final_text = self.url_buffer + final_text
            self.url_buffer = ""
            self.in_url = False

        if final_text:
            self.full_text += final_text

        # 처리할 내용 없으면 빈 문자열 반환
        if not final_text and not self.full_text:
            return ""

        try:
            # 언어 설정 가져오기
            _, _, reference_word = await self.response_generator.get_translation_language_word(
                self.request.chat.lang
            )

            # 전체 텍스트에 대한 최종 처리 수행
            processed_text = self.full_text

            # 1. 참조 추가
            if self.request.meta.rag_sys_info in await self.response_generator._get_cached_setting('source_rag_target'):
                processed_text = await self.response_generator.make_answer_reference(
                    processed_text,
                    self.request.meta.rag_sys_info,
                    reference_word,
                    self.documents,
                    self.request
                )

            # 2. VOC 처리
            if "komico_voc" in self.request.meta.rag_sys_info.split(','):
                processed_text = await self.voc_processor.make_komico_voc_groupware_docid_url(
                    processed_text
                )

            self.logger.debug(f"[{session_id}] 응답 최종 처리 완료")
            return processed_text

        except Exception as e:
            self.logger.error(f"[{session_id}] 응답 최종 처리 중 오류: {str(e)}", exc_info=True)
            # 오류 시 원본 반환
            return self.full_text

    def get_full_text(self) -> str:
        """
                전체 응답 텍스트 반환

                Returns:
                    str: 저장된 전체 텍스트
                """
        # URL 버퍼에 남은 내용도 포함
        if self.url_buffer:
            return self.full_text + self.url_buffer
        return self.full_text
