"""
vLLM 모델 전용 히스토리 핸들러 모듈

vLLM 모델을 위한 대화 히스토리 처리 기능을 제공합니다.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple

from langchain_core.documents import Document

from src.schema.chat_req import ChatRequest
from src.schema.vllm_inquery import VllmInquery
from src.services.history.handlers.history_handler import BaseHistoryHandler
from src.services.history.utils.validators import validate_rewritten_question, extract_important_entities
from src.utils.history_prompt_manager import PromptManager
from src.common.config_loader import ConfigLoader

# 설정 로드
settings = ConfigLoader().get_settings()

# 로거 설정
logger = logging.getLogger(__name__)


class VLLMHistoryHandler(BaseHistoryHandler):
    """
    vLLM 모델 전용 히스토리 핸들러 클래스

    vLLM 모델에 최적화된 대화 처리 및 스트리밍 기능을 제공합니다.
    """

    async def handle_chat_with_history(
            self,
            request: ChatRequest,
            language: str,
            rag_chat_chain: Any = None
    ) -> Optional[Dict[str, Any]]:
        """
        Ollama 모델과 호환성을 위한 메서드.
        vLLM 구현에서는 실제로 handle_chat_with_history_vllm 메서드를 사용합니다.

        Args:
            request: 채팅 요청
            language: 응답 언어
            rag_chat_chain: RAG 채팅 체인 (미사용)

        Returns:
            Optional[Dict[str, Any]]: 채팅 응답 또는 None
        """
        # vLLM 메서드로 변환하여 호출
        answer, _ = await self.handle_chat_with_history_vllm(request, language)
        return {"answer": answer} if answer else None

    async def handle_chat_with_history_vllm(
            self,
            request: ChatRequest,
            language: str
    ) -> Tuple[str, List[Document]]:
        """
        vLLM을 사용하여 히스토리 기반 채팅 요청을 처리합니다.
        개선된 2단계 접근법으로 대화 이력을 처리합니다.

        Args:
            request: 채팅 요청
            language: 응답 언어

        Returns:
            Tuple[str, List[Document]]: (응답 텍스트, 검색된 문서 목록)
        """
        session_id = self.current_session_id
        logger.debug(f"[{session_id}] vLLM 히스토리 처리 시작")

        # 개선된 질문 재작성 및 검색 사용 여부 확인
        use_improved_history = getattr(settings.llm, 'use_improved_history', False)

        if use_improved_history:
            # 개선된 2단계 접근법 사용
            return await self._handle_chat_with_history_improved(request, language)
        else:
            # 기존 방식 사용
            return await self._handle_chat_with_history_standard(request, language)

    async def _handle_chat_with_history_standard(
            self,
            request: ChatRequest,
            language: str
    ) -> Tuple[str, List[Document]]:
        """
        표준 vLLM 처리 접근법을 사용합니다.
        히스토리를 단순 포맷팅하여 컨텍스트로 제공합니다.

        Args:
            request: 채팅 요청
            language: 응답 언어

        Returns:
            Tuple[str, List[Document]]: (응답 텍스트, 검색된 문서 목록)
        """
        session_id = self.current_session_id

        # RAG 프롬프트 템플릿 가져오기
        rag_prompt_template = self._get_rag_prompt_template()

        # 검색 문서 가져오기
        logger.debug(f"[{session_id}] 문서 검색 시작")
        retrieval_document = await self.retriever.ainvoke(request.chat.user)
        logger.debug(f"[{session_id}] 검색된 문서 수: {len(retrieval_document)}")

        # 대화 히스토리 가져오기 및 포맷팅
        session_history = self.get_session_history()
        formatted_history = self.formatter.format_history_for_prompt(session_history, settings.llm.max_history_turns)
        logger.debug(f"[{session_id}] 처리된 히스토리 메시지 수: {len(session_history.messages)}")

        # 컨텍스트 준비
        common_input = {
            "input": request.chat.user,
            "history": formatted_history,
            "context": retrieval_document,
            "language": language,
            "today": self._get_today(),
        }

        # VOC 관련 설정 추가 (필요한 경우)
        if self.request.meta.rag_sys_info == "komico_voc":
            common_input.update({
                "gw_doc_id_prefix_url": settings.voc.gw_doc_id_prefix_url,
                "check_gw_word_link": settings.voc.check_gw_word_link,
                "check_gw_word": settings.voc.check_gw_word,
                "check_block_line": settings.voc.check_block_line,
            })

        # 시스템 프롬프트 생성
        vllm_inquery_context = self.formatter.build_system_prompt(rag_prompt_template, common_input)

        # vLLM 요청 생성
        vllm_request = VllmInquery(
            request_id=session_id,
            prompt=vllm_inquery_context
        )

        # vLLM 엔드포인트 호출
        logger.debug(f"[{session_id}] vLLM 엔드포인트 호출")
        response = await self.call_vllm_endpoint(vllm_request)

        # 응답 추출
        answer = response.get("generated_text", "") or response.get("answer", "")
        logger.debug(f"[{session_id}] vLLM 응답 수신, 길이: {len(answer)}")

        return answer, retrieval_document

    async def _handle_chat_with_history_improved(
            self,
            request: ChatRequest,
            language: str
    ) -> Tuple[str, List[Document]]:
        """
        개선된 vLLM 처리 접근법을 사용합니다.
        2단계로 처리: 1) 대화 이력 기반 질문 재작성, 2) 검색 및 응답 생성

        Args:
            request: 채팅 요청
            language: 응답 언어

        Returns:
            Tuple[str, List[Document]]: (응답 텍스트, 검색된 문서 목록)
        """
        session_id = self.current_session_id
        logger.debug(f"[{session_id}] 개선된 vLLM 히스토리 처리 시작")

        # 1단계: 대화 이력을 사용하여 독립적인 질문 생성
        # 대화 이력 가져오기
        session_history = self.get_session_history()
        formatted_history = self.formatter.format_history_for_prompt(session_history, settings.llm.max_history_turns)

        # 대화 이력이 없는 경우 바로 원래 질문 사용
        if not formatted_history or len(session_history.messages) == 0:
            logger.debug(f"[{session_id}] 대화 이력이 없어 원래 질문을 사용합니다.")
            rewritten_question = request.chat.user
        else:
            # 질문 재정의를 위한 프롬프트 템플릿
            rewrite_prompt_template = PromptManager.get_rewrite_prompt_template()

            # 질문 재정의 프롬프트 생성
            rewrite_context = {
                "history": formatted_history,
                "input": request.chat.user,
            }

            rewrite_prompt = rewrite_prompt_template.format(**rewrite_context)

            # vLLM에 질문 재정의 요청
            rewrite_request = VllmInquery(
                request_id=f"{session_id}_rewrite",
                prompt=rewrite_prompt
            )

            rewrite_start_time = time.time()
            logger.debug(f"[{session_id}] 질문 재정의 vLLM 요청 전송")

            try:
                # 질문 재정의 요청에 타임아웃 적용 (최대 3초)
                rewrite_timeout = getattr(settings.llm, 'rewrite_timeout', 3.0)
                rewrite_response = await asyncio.wait_for(
                    self.call_vllm_endpoint(rewrite_request),
                    timeout=rewrite_timeout
                )
                rewritten_question = rewrite_response.get("generated_text", "").strip()

                rewrite_time = time.time() - rewrite_start_time
                logger.debug(f"[{session_id}] 질문 재정의 완료: {rewrite_time:.4f}초 소요")

                # 재작성된 질문이 없거나 오류 발생 시 원래 질문 사용
                if not rewritten_question or len(rewritten_question) < 5:
                    logger.warning(f"[{session_id}] 질문 재정의 실패, 원래 질문 사용")
                    rewritten_question = request.chat.user
                else:
                    logger.debug(f"[{session_id}] 최종 재정의 질문: '{rewritten_question}'")

            except asyncio.TimeoutError:
                logger.warning(f"[{session_id}] 질문 재정의 타임아웃, 원래 질문 사용")
                rewritten_question = request.chat.user
            except Exception as e:
                logger.error(f"[{session_id}] 질문 재정의 오류: {str(e)}")
                rewritten_question = request.chat.user

        # 2단계: 재정의된 질문으로 문서 검색
        logger.debug(f"[{session_id}] 재정의된 질문으로 문서 검색 시작")
        retrieval_document = await self.retriever.ainvoke(rewritten_question)
        logger.debug(f"[{session_id}] 검색된 문서 수: {len(retrieval_document)}")

        # 3단계: 최종 응답 생성
        # RAG 프롬프트 템플릿 가져오기
        rag_prompt_template = self._get_rag_prompt_template()

        # 최종 응답 생성을 위한 컨텍스트 준비
        final_prompt_context = {
            "input": request.chat.user,  # 원래 질문 사용
            "rewritten_question": rewritten_question,  # 재작성된 질문도 제공
            "history": formatted_history,  # 형식화된 대화 이력
            "context": retrieval_document,  # 검색된 문서
            "language": language,
            "today": self._get_today(),
        }

        # VOC 관련 설정 추가 (필요한 경우)
        if self.request.meta.rag_sys_info == "komico_voc":
            final_prompt_context.update({
                "gw_doc_id_prefix_url": settings.voc.gw_doc_id_prefix_url,
                "check_gw_word_link": settings.voc.check_gw_word_link,
                "check_gw_word": settings.voc.check_gw_word,
                "check_block_line": settings.voc.check_block_line,
            })

        # 최종 시스템 프롬프트 생성
        vllm_inquery_context = self.formatter.build_system_prompt_improved(rag_prompt_template, final_prompt_context)

        # vLLM에 최종 응답 요청
        vllm_request = VllmInquery(
            request_id=session_id,
            prompt=vllm_inquery_context
        )

        logger.debug(f"[{session_id}] 최종 응답 생성 vLLM 요청 전송")
        response = await self.call_vllm_endpoint(vllm_request)
        answer = response.get("generated_text", "") or response.get("answer", "")
        logger.debug(f"[{session_id}] 최종 응답 생성 완료, 응답 길이: {len(answer)}")

        return answer, retrieval_document

    async def handle_chat_with_history_vllm_streaming(
            self,
            request: ChatRequest,
            language: str
    ) -> Tuple[VllmInquery, List[Document]]:
        """
        vLLM을 사용하여 스트리밍 모드로 히스토리 기반 채팅 요청을 처리합니다.

        Args:
            request: 채팅 요청
            language: 응답 언어

        Returns:
            Tuple[VllmInquery, List[Document]]: (vLLM 요청 객체, 검색된 문서 목록)
        """
        session_id = self.current_session_id
        logger.debug(f"[{session_id}] vLLM 스트리밍 히스토리 처리 시작")

        # 개선된 히스토리 처리 사용 여부 확인
        use_improved_history = getattr(settings.llm, 'use_improved_history', False)

        if use_improved_history:
            # 개선된 2단계 접근법 사용
            return await self._handle_streaming_improved(request, language)
        else:
            # 기존 방식 사용
            return await self._handle_streaming_standard(request, language)

    async def _handle_streaming_standard(
            self,
            request: ChatRequest,
            language: str
    ) -> Tuple[VllmInquery, List[Document]]:
        """
        표준 스트리밍 처리 접근법을 사용합니다.

        Args:
            request: 채팅 요청
            language: 응답 언어

        Returns:
            Tuple[VllmInquery, List[Document]]: (vLLM 요청 객체, 검색된 문서 목록)
        """
        session_id = self.current_session_id

        # RAG 프롬프트 템플릿 가져오기
        rag_prompt_template = self._get_rag_prompt_template()

        # 검색 문서 가져오기
        logger.debug(f"[{session_id}] 스트리밍용 검색 문서 가져오기 시작")
        retrieval_document = await self.retriever.ainvoke(request.chat.user)
        logger.debug(f"[{session_id}] 스트리밍용 검색 문서 가져오기 완료: {len(retrieval_document)}개")

        # 채팅 히스토리 가져오기 및 최적화
        session_history = self.get_session_history()
        formatted_history = self.formatter.format_history_for_prompt(session_history, settings.llm.max_history_turns)
        logger.debug(f"[{session_id}] 스트리밍용 {len(session_history.messages)}개 히스토리 메시지 처리 완료")

        # 컨텍스트 준비
        common_input = {
            "input": request.chat.user,
            "history": formatted_history,  # 최적화된 형식 사용
            "context": retrieval_document,
            "language": language,
            "today": self._get_today(),
        }

        # VOC 관련 설정 추가 (필요한 경우)
        if self.request.meta.rag_sys_info == "komico_voc":
            common_input.update({
                "gw_doc_id_prefix_url": settings.voc.gw_doc_id_prefix_url,
                "check_gw_word_link": settings.voc.check_gw_word_link,
                "check_gw_word": settings.voc.check_gw_word,
                "check_block_line": settings.voc.check_block_line,
            })

        # 시스템 프롬프트 생성
        vllm_inquery_context = self.formatter.build_system_prompt(rag_prompt_template, common_input)

        # 스트리밍을 위한 vLLM 요청 생성
        vllm_request = VllmInquery(
            request_id=session_id,
            prompt=vllm_inquery_context,
            stream=True  # 스트리밍 모드 활성화
        )

        logger.debug(f"[{session_id}] 스트리밍 요청 준비 완료")
        return vllm_request, retrieval_document

    async def _handle_streaming_improved(
            self,
            request: ChatRequest,
            language: str
    ) -> Tuple[VllmInquery, List[Document]]:
        """
        개선된 스트리밍 처리 접근법을 사용합니다.
        2단계로 처리: 1) 대화 이력 기반 질문 재작성, 2) 검색 및 스트리밍 설정

        Args:
            request: 채팅 요청
            language: 응답 언어

        Returns:
            Tuple[VllmInquery, List[Document]]: (vLLM 요청 객체, 검색된 문서 목록)
        """
        session_id = self.current_session_id
        logger.debug(f"[{session_id}] 개선된 vLLM 스트리밍 히스토리 처리 시작")

        # 1단계: 대화 이력을 사용하여 독립적인 질문 생성
        # 대화 이력 가져오기
        session_history = self.get_session_history()
        formatted_history = self.formatter.format_history_for_prompt(session_history, settings.llm.max_history_turns)

        # 대화 이력이 없는 경우 바로 원래 질문 사용
        if not formatted_history or len(session_history.messages) == 0:
            logger.debug(f"[{session_id}] 대화 이력이 없어 원래 질문을 사용합니다.")
            rewritten_question = request.chat.user
        else:
            # 질문 재정의를 위한 프롬프트 템플릿
            rewrite_prompt_template = PromptManager.get_rewrite_prompt_template()

            # 질문 재정의 프롬프트 생성
            rewrite_context = {
                "history": formatted_history,
                "input": request.chat.user,
            }

            rewrite_prompt = rewrite_prompt_template.format(**rewrite_context)

            # vLLM에 질문 재정의 요청
            rewrite_request = VllmInquery(
                request_id=f"{session_id}_rewrite",
                prompt=rewrite_prompt
            )

            logger.debug(f"[{session_id}] 질문 재정의 vLLM 요청 전송")
            try:
                # 질문 재정의 요청에 타임아웃 적용 (최대 3초)
                rewrite_timeout = getattr(settings.llm, 'rewrite_timeout', 3.0)
                rewrite_response = await asyncio.wait_for(
                    self.call_vllm_endpoint(rewrite_request),
                    timeout=rewrite_timeout
                )
                rewritten_question = rewrite_response.get("generated_text", "").strip()

                # 재작성된 질문이 없거나 오류 발생 시 원래 질문 사용
                if not rewritten_question or len(rewritten_question) < 5:
                    logger.warning(f"[{session_id}] 질문 재정의 실패, 원래 질문 사용")
                    rewritten_question = request.chat.user
                else:
                    logger.debug(f"[{session_id}] 최종 재정의 질문: '{rewritten_question}'")
            except asyncio.TimeoutError:
                logger.warning(f"[{session_id}] 질문 재정의 타임아웃, 원래 질문 사용")
                rewritten_question = request.chat.user
            except Exception as e:
                logger.error(f"[{session_id}] 질문 재정의 오류: {str(e)}")
                rewritten_question = request.chat.user

        # 2단계: 재정의된 질문으로 문서 검색
        logger.debug(f"[{session_id}] 재정의된 질문으로 문서 검색 시작")
        retrieval_document = []  # 기본값으로 빈 리스트 설정

        try:
            # 검색기가 초기화되어 있는지 확인 (오류 방지)
            if self.retriever is not None:
                retrieval_document = await self.retriever.ainvoke(rewritten_question)
                logger.debug(f"[{session_id}] 문서 검색 완료: {len(retrieval_document)}개 문서")
            else:
                logger.warning(f"[{session_id}] 검색기가 초기화되지 않았습니다. 빈 문서 리스트 사용")
        except Exception as e:
            logger.error(f"[{session_id}] 문서 검색 중 오류: {str(e)}")

        # 3단계: 최종 스트리밍 응답 준비
        # RAG 프롬프트 템플릿 가져오기
        rag_prompt_template = self._get_rag_prompt_template()

        # 최종 응답 생성을 위한 컨텍스트 준비
        final_prompt_context = {
            "input": request.chat.user,  # 원래 질문 사용
            "rewritten_question": rewritten_question,  # 재작성된 질문도 제공
            "history": formatted_history,  # 형식화된 대화 이력
            "context": retrieval_document,  # 검색된 문서
            "language": language,
            "today": self._get_today(),
        }

        # VOC 관련 설정 추가 (필요한 경우)
        if request.meta.rag_sys_info == "komico_voc":
            final_prompt_context.update({
                "gw_doc_id_prefix_url": settings.voc.gw_doc_id_prefix_url,
                "check_gw_word_link": settings.voc.check_gw_word_link,
                "check_gw_word": settings.voc.check_gw_word,
                "check_block_line": settings.voc.check_block_line,
            })

        # 개선된 시스템 프롬프트 빌드 함수 사용
        vllm_inquery_context = self.formatter.build_system_prompt_improved(rag_prompt_template, final_prompt_context)

        # 스트리밍을 위한 vLLM 요청 생성
        vllm_request = VllmInquery(
            request_id=session_id,
            prompt=vllm_inquery_context,
            stream=True  # 스트리밍 모드 활성화
        )

        logger.debug(
            f"[{session_id}] 스트리밍 요청 준비 완료 - "
            f"원본 질문: '{request.chat.user}', "
            f"재작성된 질문: '{rewritten_question}', "
            f"검색된 문서 수: {len(retrieval_document)}"
        )

        # 성능 개선을 위한 통계 측정
        self.response_stats = {
            "rewrite_time": 0,
            "retrieval_time": 0,
            "total_prep_time": time.time()
        }

        return vllm_request, retrieval_document

    def _get_rag_prompt_template(self) -> str:
        """
        RAG 프롬프트 템플릿을 가져옵니다.

        Returns:
            str: RAG 프롬프트 템플릿
        """
        from src.common.config_loader import ConfigLoader
        from src.common.query_check_dict import QueryCheckDict
        from src.services.response_generator import ResponseGenerator

        # 설정 및 프롬프트 정보 로드
        app_settings = ConfigLoader().get_settings()
        query_check_dict = QueryCheckDict(app_settings.prompt.llm_prompt_path)
        response_generator = ResponseGenerator(app_settings, query_check_dict)

        # RAG 시스템에 맞는 프롬프트 템플릿 가져오기
        return response_generator.get_rag_qa_prompt(self.current_rag_sys_info)

    @classmethod
    def _get_today(cls) -> str:
        """
        현재 날짜와 요일을 한국어 형식으로 반환합니다.

        Returns:
            str: 포맷된 날짜 문자열
        """
        from datetime import datetime
        import time

        # 요일 이름 정의
        day_names = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']

        # 현재 날짜/시간 가져오기
        today = datetime.now()
        weekday = day_names[time.localtime().tm_wday]

        # 한국어 형식으로 포맷팅
        return f"{today.strftime('%Y년 %m월 %d일')} {weekday} {today.strftime('%H시 %M분')}입니다."
