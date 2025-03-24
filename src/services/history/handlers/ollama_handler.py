"""
Ollama 모델 전용 히스토리 핸들러 모듈

Ollama 모델을 위한 대화 히스토리 처리 기능을 제공합니다.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory

from src.schema.chat_req import ChatRequest
from src.services.history.handlers.history_handler import BaseHistoryHandler
from src.utils.prompt import PromptManager

# 로거 설정
logger = logging.getLogger(__name__)


class OllamaHistoryHandler(BaseHistoryHandler):
    """
    Ollama 모델 전용 히스토리 핸들러 클래스

    Ollama 모델에 최적화된 체인 구성 및 대화 처리 기능을 제공합니다.
    """

    def init_chat_chain_with_history(self):
        """
        히스토리가 포함된 채팅 체인을 초기화하거나 캐시에서 가져옵니다.

        Returns:
            Any: 초기화된 채팅 체인
        """
        cache_key = f"{self.current_rag_sys_info}:{self.current_session_id}"

        # 캐시에서 체인 확인
        cached_chain = self.cache_manager.get("chain", cache_key)
        if cached_chain and self._is_chain_valid(cached_chain):
            logger.debug(f"[{self.current_session_id}] 캐시된 채팅 체인 사용")
            return cached_chain

        # 새 체인 생성
        logger.debug(f"[{self.current_session_id}] 새 Ollama 채팅 체인 생성")

        # RAG 프롬프트 템플릿 가져오기
        rag_prompt_template = self._get_rag_prompt_template()

        # 채팅 체인 프롬프트 구성
        rag_chain_prompt = ChatPromptTemplate.from_messages([
            ("system", rag_prompt_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        # 히스토리 인식 검색기 생성
        history_aware_retriever = create_history_aware_retriever(
            self.llm_model,
            self.retriever,
            PromptManager.get_contextualize_q_prompt()
        )

        # 질문-응답 체인 생성
        question_answer_chain = create_stuff_documents_chain(
            self.llm_model,
            rag_chain_prompt
        )

        # 검색 체인 생성
        chat_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain
        )

        # 체인 유효성 검사
        if not self._is_chain_valid(chat_chain):
            logger.error(f"[{self.current_session_id}] 새로 생성된 체인이 유효하지 않습니다.")
            raise ValueError("체인 생성 실패")

        # 캐시에 체인 저장
        self.cache_manager.set("chain", cache_key, chat_chain)

        return chat_chain

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
        settings = ConfigLoader().get_settings()
        query_check_dict = QueryCheckDict(settings.prompt.llm_prompt_path)
        response_generator = ResponseGenerator(settings, query_check_dict)

        # RAG 시스템에 맞는 프롬프트 템플릿 가져오기
        return response_generator.get_rag_qa_prompt(self.current_rag_sys_info)

    @staticmethod
    def _is_chain_valid(chain) -> bool:
        """
        체인이 호출 가능한지 확인합니다.

        Args:
            chain: 검증할 체인

        Returns:
            bool: 체인이 유효하면 True, 아니면 False
        """
        return hasattr(chain, 'invoke') and callable(getattr(chain, 'invoke'))

    async def handle_chat_with_history(
            self,
            request: ChatRequest,
            language: str,
            rag_chat_chain: Any
    ) -> Optional[Dict[str, Any]]:
        """
        Ollama 모델로 히스토리를 활용한 채팅 요청을 처리합니다.

        Args:
            request: 채팅 요청
            language: 응답 언어
            rag_chat_chain: RAG 채팅 체인

        Returns:
            Optional[Dict[str, Any]]: 채팅 응답 또는 None
        """
        session_id = self.current_session_id

        # 히스토리 관리를 포함한 체인 구성
        conversational_rag_chain = RunnableWithMessageHistory(
            runnable=rag_chat_chain,
            get_session_history=self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_message_key="answer",
        )

        # 공통 입력 준비
        common_input = {
            "original_question": request.chat.user,
            "input": request.chat.user,
            "history": "",  # Ollama 처리용 빈 문자열
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

        # 중복 처리 방지를 위한 입력 ID
        input_id = (common_input["original_question"], session_id)

        # 이미 처리된 입력인지 확인
        if self.cache_manager.is_in_processed_set("inputs", input_id):
            logger.warning(f"[{session_id}] 중복 입력 감지됨, 빈 응답 반환")
            return {"answer": ""}

        try:
            # 처리 중인 입력으로 표시
            self.cache_manager.add_to_processed_set("inputs", input_id)

            logger.debug(f"[{session_id}] Ollama RAG 체인 호출 시작")

            # 체인 호출 방식 결정 (동기/비동기)
            if hasattr(conversational_rag_chain, "ainvoke") and callable(getattr(conversational_rag_chain, "ainvoke")):
                response = await conversational_rag_chain.ainvoke(common_input)
            else:
                logger.error(f"[{session_id}] 제공된 체인에 유효한 'ainvoke' 메서드가 없습니다.")
                return None

            logger.debug(f"[{session_id}] Ollama RAG 체인 호출 완료")

            # 응답 유효성 검사
            if not response:
                logger.error(f"[{session_id}] 체인 응답이 비어 있거나 None입니다.")
                return {"answer": "모델에서 응답이 없습니다."}

            return response

        except Exception as e:
            logger.error(f"[{session_id}] 체인 호출 중 오류 발생: {e}")
            return None
        finally:
            # 처리 완료된 입력 표시 해제
            if self.cache_manager.is_in_processed_set("inputs", input_id):
                self.cache_manager.remove_from_processed_set("inputs", input_id)

    async def handle_chat_with_history_vllm(
            self,
            request: ChatRequest,
            language: str
    ) -> Tuple[str, List[Document]]:
        """
        vLLM을 사용하여 히스토리 기반 채팅 요청을 처리합니다.
        이 메서드는 Ollama 핸들러에서는 지원되지 않습니다.

        Args:
            request: 채팅 요청
            language: 응답 언어

        Returns:
            Tuple[str, List[Document]]: (응답 텍스트, 검색된 문서 목록)
        """
        logger.warning(f"[{self.current_session_id}] Ollama 핸들러에서는 vLLM 기능이 지원되지 않습니다.")
        return "Ollama 핸들러에서는 vLLM 기능이 지원되지 않습니다.", []

    async def handle_chat_with_history_vllm_streaming(
            self,
            request: ChatRequest,
            language: str
    ) -> Tuple[Any, List[Document]]:
        """
        vLLM을 사용하여 스트리밍 모드로 히스토리 기반 채팅 요청을 처리합니다.
        이 메서드는 Ollama 핸들러에서는 지원되지 않습니다.

        Args:
            request: 채팅 요청
            language: 응답 언어

        Returns:
            Tuple[Any, List[Document]]: (요청 객체, 문서 목록)
        """
        logger.warning(f"[{self.current_session_id}] Ollama 핸들러에서는 vLLM 스트리밍 기능이 지원되지 않습니다.")
        return None, []

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
