"""
채팅 이력 처리 모듈
===============

대화 이력 관리 및 컨텍스트 유지 기능을 제공합니다.

기능:
- 채팅 이력 저장 및 관리
- 대화 컨텍스트 기반 검색
- 스트리밍 응답 지원
"""

import asyncio
import logging
import re
import time
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Set

from cachetools import TTLCache
from dateutil.parser import parse
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory

from src.schema.chat_req import ChatRequest
from src.schema.vllm_inquery import VllmInquery
from src.services.retrieval.base import RetrievalSourceBase
from src.services.response.base import HistoryHandlerBase
from src.services.response.factory import HistoryHandlerFactory
from src.utils.redis_utils import RedisUtils
from src.utils.prompt import PromptManager

# 로거 설정
logger = logging.getLogger(__name__)


class StandardHistoryHandler(HistoryHandlerBase):
    """
    표준 채팅 이력 처리 구현

    대화 이력을 관리하고 이를 활용하여 컨텍스트 기반 응답을 생성합니다.
    """

    # 세션 캐시
    session_cache = TTLCache(maxsize=100, ttl=3600)  # 1시간 캐시
    session_chain_cache = TTLCache(maxsize=50, ttl=3600)  # 1시간 캐시
    chain_lock = Lock()

    def __init__(self, llm_model: Any, request: ChatRequest, max_history_turns: int = 10):
        """
        이력 처리기 초기화

        Args:
            llm_model: LLM 모델
            request: 채팅 요청 객체
            max_history_turns: 유지할 최대 대화 턴 수
        """
        super().__init__(request, max_history_turns)
        self.llm_model = llm_model
        self.current_session_id = request.meta.session_id
        self.current_rag_sys_info = request.meta.rag_sys_info
        self.request = request
        self.max_history_turns = max_history_turns

        self.session_lock = Lock()
        self.processed_inputs = set()
        self.processed_message_ids = set()  # 중복 방지를 위한 메시지 ID 추적
        self.retrieval_source: Optional[RetrievalSourceBase] = None

        # 성능 통계
        self.response_stats = None

    async def initialize(self) -> bool:
        """
        이력 처리기 초기화

        Returns:
            bool: 초기화 성공 여부
        """
        # 특별한 초기화 작업 없음
        return True

    @classmethod
    def is_gemma_model(cls, settings) -> bool:
        """
        현재 모델이 Gemma 모델인지 확인

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

    @classmethod
    def build_system_prompt_gemma(cls, template: str, context: Dict[str, Any]) -> str:
        """
        Gemma에 맞는 형식으로 시스템 프롬프트를 구성

        Args:
            template: 프롬프트 템플릿
            context: 템플릿에 적용할 변수들

        Returns:
            str: Gemma 형식의 시스템 프롬프트
        """
        try:
            # 먼저 기존 format 메서드로 변수를 대체
            raw_prompt = template.format(**context)

            # Gemma 형식으로 변환
            # <start_of_turn>user 형식으로 시작
            formatted_prompt = "<start_of_turn>user\n"

            # 시스템 프롬프트 삽입
            formatted_prompt += raw_prompt

            # 사용자 입력부 종료 및 모델 응답 시작
            formatted_prompt += "\n<end_of_turn>\n<start_of_turn>model\n"

            return formatted_prompt

        except KeyError as e:
            # 누락된 키 처리
            missing_key = str(e).strip("'")
            logger.warning(f"시스템 프롬프트 템플릿에 키가 누락됨: {missing_key}, 빈 문자열로 대체합니다.")
            context[missing_key] = ""
            return cls.build_system_prompt_gemma(template, context)
        except Exception as e:
            # 기타 예외 처리
            logger.error(f"Gemma 시스템 프롬프트 형식화 중 오류: {e}")
            # 기본 Gemma 프롬프트로 폴백
            basic_prompt = (f"<start_of_turn>user\n다음 질문에 답해주세요: {context.get('input', '질문 없음')}\n<end_of_turn>\n"
                            f"<start_of_turn>model\n")
            return basic_prompt

    async def init_retrieval_source(self, retrieval_source: RetrievalSourceBase) -> None:
        """
        검색 소스 초기화

        Args:
            retrieval_source: 검색 소스 인스턴스
        """
        self.retrieval_source = retrieval_source
        logger.debug(f"[{self.current_session_id}] 검색 소스 설정됨")

    async def init_chat_chain_with_history(self):
        """
        이력 기반 채팅 체인 초기화 또는 캐시된 체인 반환

        Returns:
            Any: 초기화된 체인 또는 캐시된 체인
        """
        cache_key = f"{self.current_rag_sys_info}:{self.current_session_id}"

        with self.chain_lock:
            if cache_key in self.session_chain_cache:
                cached_chain = self.session_chain_cache[cache_key]
                if self.is_chain_valid(cached_chain):
                    logger.debug(f"[{self.current_session_id}] 캐시된 채팅 체인 사용")
                    return cached_chain
                logger.warning(f"[{self.current_session_id}] 캐시된 체인이 유효하지 않습니다. 체인을 재생성합니다.")

            # 검색 소스 확인
            if not self.retrieval_source:
                logger.error(f"[{self.current_session_id}] 검색 소스가 초기화되지 않았습니다.")
                raise ValueError("검색 소스가 초기화되지 않았습니다.")

            # 프롬프트 템플릿 가져오기
            from src.services.response.generator import ResponseGenerator
            from src.common.query_check_dict import QueryCheckDict
            from src.common.config_loader import ConfigLoader

            settings = ConfigLoader().get_settings()
            query_check_dict = QueryCheckDict(settings.prompt.llm_prompt_path)
            response_generator = ResponseGenerator(settings, query_check_dict)

            rag_prompt_template = response_generator.get_rag_qa_prompt(self.current_rag_sys_info)
            rag_chain_prompt = ChatPromptTemplate.from_messages([
                ("system", rag_prompt_template),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ])

            # 이력 기반 검색기 생성
            history_aware_retriever = create_history_aware_retriever(
                self.llm_model,
                self.retrieval_source,
                PromptManager.get_contextualize_q_prompt()
            )

            # 질의응답 체인 생성
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
            if not self.is_chain_valid(chat_chain):
                logger.error(f"[{self.current_session_id}] 새로 생성된 체인이 유효하지 않습니다.")
                raise ValueError("체인 생성 실패.")

            # 캐시에 저장
            self.session_chain_cache[cache_key] = chat_chain
            logger.debug(f"[{self.current_session_id}] 새 채팅 체인 생성 및 캐싱 완료")
            return chat_chain

    @staticmethod
    def is_chain_valid(chain) -> bool:
        """
        체인의 유효성 검사

        Args:
            chain: 검사할 체인 객체

        Returns:
            bool: 유효하면 True, 아니면 False
        """
        return hasattr(chain, 'invoke') and callable(getattr(chain, 'invoke'))

    async def get_session_history(self) -> ChatMessageHistory:
        """
        세션 이력 가져오기 또는 초기화

        메시지를 타임스탬프로 정렬하고 중복을 제거합니다.

        Returns:
            ChatMessageHistory: 세션 이력
        """
        if not self.current_session_id:
            raise ValueError("세션 ID가 유효하지 않습니다.")

        cache_key = f"{self.current_rag_sys_info}:{self.current_session_id}"

        with self.session_lock:
            # 캐시된 세션 이력 가져오기 또는 새 이력 초기화
            session_history = self.session_cache.get(cache_key, ChatMessageHistory())

            try:
                # Redis에서 메시지 가져오기
                history_messages = await RedisUtils.async_get_messages_from_redis(
                    self.current_rag_sys_info, self.current_session_id
                )
            except Exception as e:
                logger.error(f"[{self.current_session_id}] Redis에서 메시지 가져오기 실패: {e}")
                history_messages = []

            # 타임스탬프 정보가 있는 메시지 처리
            all_messages = []
            for entry in history_messages:
                for msg in entry.get("messages", []):
                    # 고유 메시지 식별자 생성
                    msg_id = f"{msg['content']}_{msg['timestamp']}"

                    # 중복 메시지 건너뛰기
                    if msg_id not in self.processed_message_ids:
                        try:
                            timestamp = parse(msg["timestamp"])
                            all_messages.append((timestamp, msg))
                            self.processed_message_ids.add(msg_id)
                        except Exception as e:
                            logger.error(f"[{self.current_session_id}] 타임스탬프 파싱 오류: {e}")

            # 타임스탬프로 메시지 정렬 (오름차순 - 가장 오래된 것부터)
            all_messages.sort(key=lambda x: x[0])

            # 가장 최근 대화로 이력 제한
            max_messages = self.max_history_turns * 2  # 각 턴은 사용자 메시지와 응답으로 구성
            if len(all_messages) > max_messages:
                logger.debug(f"[{self.current_session_id}] 이력을 최근 {max_messages}개 메시지로 제한")
                all_messages = all_messages[-max_messages:]

            # 중복 콘텐츠 메시지 제거
            all_messages = self._remove_duplicate_content_messages(all_messages)

            # 올바른 순서로 다시 빌드하기 위해 세션 이력 비우기
            session_history.clear()

            # 세션 이력에 메시지 추가
            for _, msg in all_messages:
                try:
                    if msg["role"] == "HumanMessage":
                        session_history.add_user_message(
                            HumanMessage(
                                content=msg["content"],
                                additional_kwargs={"timestamp": msg["timestamp"]}
                            )
                        )
                    elif msg["role"] == "AIMessage":
                        session_history.add_ai_message(
                            AIMessage(
                                content=msg["content"],
                                additional_kwargs={"timestamp": msg["timestamp"]}
                            )
                        )
                except Exception as e:
                    logger.error(f"[{self.current_session_id}] 메시지 추가 오류: {e}")

            # 캐시 업데이트
            self.session_cache[cache_key] = session_history

            logger.debug(
                f"[{self.current_session_id}] Redis에서 {len(session_history.messages)}개 이력 메시지 가져옴")

            return session_history

    @classmethod
    def _remove_duplicate_content_messages(
            cls,
            messages: List[Tuple[Any, Dict[str, Any]]]
    ) -> List[Tuple[Any, Dict[str, Any]]]:
        """
        중복 콘텐츠 메시지 제거 (가장 최근 것 유지)

        Args:
            messages: (타임스탬프, 메시지) 튜플 리스트

        Returns:
            List: 중복 제거된 (타임스탬프, 메시지) 튜플 리스트
        """
        unique_messages = []
        seen_content = set()

        # 메시지 역순 처리 (최신 것 먼저) - 동일 내용은 최신 것만 유지
        for msg_tuple in reversed(messages):
            timestamp, msg = msg_tuple

            # 비교를 위한 콘텐츠 정규화 (추가 공백 제거, 소문자화)
            normalized_content = ' '.join(msg["content"].lower().split())

            # 너무 긴 경우 비교를 위해 앞부분만 사용
            comparison_key = normalized_content[:200] if len(normalized_content) > 200 else normalized_content

            if comparison_key not in seen_content:
                unique_messages.append(msg_tuple)
                seen_content.add(comparison_key)

        # 원래 순서로 다시 정렬 (가장 오래된 것부터)
        unique_messages.sort(key=lambda x: x[0])

        return unique_messages

    @classmethod
    def format_history_for_prompt(cls, session_history: ChatMessageHistory, max_turns: int = 5) -> str:
        """
        형식화된 대화 이력 생성

        이전 대화의 맥락을 효과적으로 포착하기 위한 개선된 형식을 사용합니다.

        Args:
            session_history: 채팅 메시지 이력
            max_turns: 포함할 최대 대화 턴 수

        Returns:
            str: 형식화된 대화 이력 문자열
        """
        try:
            # 매개변수 유효성 검사
            if not session_history or not hasattr(session_history, 'messages'):
                logger.warning("유효하지 않은 session_history 객체가 제공되었습니다.")
                return ""

            messages = session_history.messages
            if not messages:
                return ""

            # 가장 최근 대화부터 max_turns 수만큼만 추출
            if len(messages) > max_turns * 2:  # 각 턴은 사용자 메시지와 시스템 응답을 포함
                messages = messages[-(max_turns * 2):]

            formatted_history = []

            # 대화 이력 프롬프트 헤더 추가
            formatted_history.append("# 이전 대화 내용")

            # 대화 턴 구성
            turns = []
            current_turn = {"user": None, "assistant": None}

            for msg in messages:
                # 타입 검사 추가
                if hasattr(msg, '__class__') and hasattr(msg.__class__, '__name__'):
                    msg_type = msg.__class__.__name__
                else:
                    msg_type = str(type(msg))

                # HumanMessage 처리
                if isinstance(msg, HumanMessage) or "HumanMessage" in msg_type:
                    # 이전 턴이 있으면 저장
                    if current_turn["user"] is not None and current_turn["assistant"] is not None:
                        turns.append(current_turn)
                        current_turn = {"user": None, "assistant": None}

                    # 현재 사용자 메시지 저장
                    if hasattr(msg, 'content'):
                        current_turn["user"] = msg.content
                    else:
                        # content 속성이 없는 경우 문자열 변환 시도
                        current_turn["user"] = str(msg)

                # AIMessage 처리
                elif isinstance(msg, AIMessage) or "AIMessage" in msg_type:
                    if hasattr(msg, 'content'):
                        current_turn["assistant"] = msg.content
                    else:
                        # content 속성이 없는 경우 문자열 변환 시도
                        current_turn["assistant"] = str(msg)

            # 마지막 턴 저장
            if current_turn["user"] is not None:
                turns.append(current_turn)

            # 턴 수가 많으면 가장 최근 턴 유지
            if len(turns) > max_turns:
                turns = turns[-max_turns:]

            # 형식화된 대화 이력 생성
            for i, turn in enumerate(turns):
                formatted_history.append(f"\n## 대화 {i + 1}")

                if turn["user"]:
                    formatted_history.append(f"User: {turn['user']}")

                if turn["assistant"]:
                    formatted_history.append(f"Assistant: {turn['assistant']}")

            # 개선된 프롬프트 지시문 추가
            formatted_history.append("\n# 현재 질문에 답변할 때 위 대화 내용을 참고하세요.")

            return "\n".join(formatted_history)

        except Exception as e:
            # 예외 발생 시 로깅하고 빈 문자열 반환
            logger.error(f"대화 이력 형식화 중 오류 발생: {str(e)}")
            return ""

    @classmethod
    def format_history_for_gemma(cls, session_history: ChatMessageHistory, max_turns: int = 5) -> str:
        """
        Gemma 모델에 적합한 형식으로 대화 이력을 구성

        Gemma의 <start_of_turn>user/<start_of_turn>model 형식을 사용합니다.

        Args:
            session_history: 채팅 메시지 이력
            max_turns: 포함할 최대 대화 턴 수

        Returns:
            str: Gemma 형식의 대화 이력 문자열
        """
        try:
            # 매개변수 유효성 검사
            if not session_history or not hasattr(session_history, 'messages'):
                logger.warning("유효하지 않은 session_history 객체가 제공되었습니다.")
                return ""

            messages = session_history.messages
            if not messages:
                return ""

            # 가장 최근 대화부터 max_turns 수만큼만 추출
            if len(messages) > max_turns * 2:  # 각 턴은 사용자 메시지와 시스템 응답을 포함
                messages = messages[-(max_turns * 2):]

            formatted_history = []

            # 대화 턴 구성
            for i in range(0, len(messages), 2):
                # 사용자 메시지
                if i < len(messages):
                    user_msg = messages[i]
                    if hasattr(user_msg, 'content'):
                        formatted_history.append(f"<start_of_turn>user\n{user_msg.content}<end_of_turn>")

                # 시스템 메시지
                if i + 1 < len(messages):
                    sys_msg = messages[i + 1]
                    if hasattr(sys_msg, 'content'):
                        formatted_history.append(f"<start_of_turn>model\n{sys_msg.content}<end_of_turn>")

            return "\n".join(formatted_history)

        except Exception as e:
            # 예외 발생 시 로깅하고 빈 문자열 반환
            logger.error(f"Gemma 대화 이력 형식화 중 오류 발생: {str(e)}")
            return ""

    async def handle_chat_with_history(self, request: ChatRequest, language: str) -> Tuple[str, List[Document]]:
        """
        이력을 활용한 채팅 처리

        Args:
            request: 채팅 요청 객체
            language: 응답 언어

        Returns:
            Tuple[str, List[Document]]: (응답 텍스트, 검색된 문서)
        """
        # 모델 유형에 따라 적절한 방식으로 처리
        if self.is_gemma_model:
            return await self.handle_chat_with_history_gemma(request, language)
        else:
            # 개선된 이력 처리 사용 여부 확인
            from src.common.config_loader import ConfigLoader
            settings = ConfigLoader().get_settings()

            if getattr(settings.llm, 'use_improved_history', False):
                return await self.handle_chat_with_history_improved(request, language)
            else:
                return await self._handle_chat_with_history_original(request, language)

    async def _handle_chat_with_history_original(self,
                                                 request: ChatRequest,
                                                 language: str) -> Tuple[str, List[Document]]:
        """
        기존 방식의 이력 기반 채팅 처리

        Args:
            request: 채팅 요청 객체
            language: 응답 언어

        Returns:
            Tuple[str, List[Document]]: (응답 텍스트, 검색된 문서)
        """
        # 이력 기반 체인 초기화
        chat_chain = await self.init_chat_chain_with_history()

        # 대화 이력 가져오기
        session_history = await self.get_session_history()

        # 공통 입력 준비
        from src.services.response.generator import ResponseGenerator
        from src.common.query_check_dict import QueryCheckDict
        from src.common.config_loader import ConfigLoader

        settings = ConfigLoader().get_settings()
        query_check_dict = QueryCheckDict(settings.prompt.llm_prompt_path)
        response_generator = ResponseGenerator(settings, query_check_dict)

        common_input = {
            "original_question": request.chat.user,
            "input": request.chat.user,
            "history": self.format_history_for_prompt(session_history),
            "language": language,
            "today": await response_generator.get_today(),
        }

        # VOC 관련 컨텍스트 추가
        if request.meta.rag_sys_info == "komico_voc":
            common_input.update({
                "gw_doc_id_prefix_url": settings.voc.gw_doc_id_prefix_url,
                "check_gw_word_link": settings.voc.check_gw_word_link,
                "check_gw_word": settings.voc.check_gw_word,
                "check_block_line": settings.voc.check_block_line,
            })

        try:
            # 이력 기반 대화 체인 구성
            conversational_rag_chain = RunnableWithMessageHistory(
                runnable=chat_chain,
                get_session_history=self.get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_message_key="answer",
            )

            # 체인 호출
            logger.debug(f"[{self.current_session_id}] 이력 기반 RAG 체인 호출")
            response = await self.invoke_chain(conversational_rag_chain, common_input)

            if not response or "answer" not in response:
                logger.error(f"[{self.current_session_id}] 응답 형식이 잘못되었습니다: {response}")
                return "응답을 생성할 수 없습니다. 다시 시도해 주세요.", []

            # 검색된 문서 추출
            context_docs = response.get("context", [])

            return response["answer"], context_docs

        except Exception as e:
            logger.error(f"[{self.current_session_id}] 이력 기반 채팅 처리 중 오류: {e}")
            return "죄송합니다. 응답 생성 중 오류가 발생했습니다.", []

    async def handle_chat_with_history_improved(self,
                                                request: ChatRequest,
                                                language: str) -> Tuple[str, List[Document]]:
        """
        개선된 2단계 접근법의 이력 기반 채팅 처리

        1. 대화 이력과 현재 질문으로 독립적인 질문 생성
        2. 독립적인 질문으로 문서 검색 및 최종 응답 생성

        Args:
            request: 채팅 요청 객체
            language: 응답 언어

        Returns:
            Tuple[str, List[Document]]: (응답 텍스트, 검색된 문서)
        """
        session_id = request.meta.session_id
        logger.debug(f"[{session_id}] 개선된 이력 처리 방식으로 채팅 시작")

        # 검색 소스 확인
        if not self.retrieval_source:
            logger.error(f"[{session_id}] 검색 소스가 초기화되지 않았습니다.")
            return "죄송합니다. 검색 소스가 초기화되지 않았습니다.", []

        # 1단계: 대화 이력을 사용하여 독립적인 질문 생성
        # 대화 이력 가져오기
        session_history = await self.get_session_history()
        formatted_history = self.format_history_for_prompt(session_history)

        # 대화 이력이 없는 경우 바로 원래 질문 사용
        if not formatted_history or len(session_history.messages) == 0:
            logger.debug(f"[{session_id}] 대화 이력이 없어 원래 질문을 사용합니다.")
            rewritten_question = request.chat.user
        else:
            # 질문 재정의를 위한 프롬프트 템플릿
            # JSON 파일에서 템플릿 불러오기
            rewrite_prompt_template = PromptManager.get_rewrite_prompt_template()

            # 질문 재정의 프롬프트 생성
            rewrite_context = {
                "history": formatted_history,
                "input": request.chat.user,
            }

            # LLM을 통한 질문 재정의
            from src.common.config_loader import ConfigLoader
            from src.services.core.llm import LLMServiceFactory

            settings = ConfigLoader().get_settings()
            llm_service = await LLMServiceFactory.create_service(settings)

            try:
                # 타임아웃 적용 (최대 3초)
                rewrite_timeout = getattr(settings.llm, 'rewrite_timeout', 3.0)
                rewrite_prompt = rewrite_prompt_template.format(**rewrite_context)

                rewritten_question = await asyncio.wait_for(
                    llm_service.ask(
                        query=request.chat.user,
                        documents=[Document(page_content=formatted_history)],
                        language=language,
                        context={"task": "rewrite_query", "prompt": rewrite_prompt}
                    ),
                    timeout=rewrite_timeout
                )

                # 재작성된 질문이 없거나 오류 발생 시 원래 질문 사용
                if not rewritten_question or len(rewritten_question) < 5:
                    logger.warning(f"[{session_id}] 질문 재정의 실패, 원래 질문 사용")
                    rewritten_question = request.chat.user
                else:
                    logger.debug(f"[{session_id}] 최종 재정의된 질문: '{rewritten_question}'")
            except asyncio.TimeoutError:
                logger.warning(f"[{session_id}] 질문 재정의 타임아웃, 원래 질문 사용")
                rewritten_question = request.chat.user
            except Exception as e:
                logger.error(f"[{session_id}] 질문 재정의 오류: {str(e)}")
                rewritten_question = request.chat.user

            # 2단계: 재정의된 질문으로 문서 검색
        logger.debug(f"[{session_id}] 재정의된 질문으로 문서 검색 시작")
        try:
            # 검색 실행
            retrieval_document = await self.retrieval_source.retrieve(rewritten_question)
            logger.debug(f"[{session_id}] 문서 검색 완료: {len(retrieval_document)}개 문서")
        except Exception as e:
            logger.error(f"[{session_id}] 문서 검색 중 오류: {str(e)}")
            retrieval_document = []

        # 3단계: 최종 응답 생성
        # RAG 프롬프트 템플릿 가져오기
        from src.services.response.generator import ResponseGenerator
        from src.common.query_check_dict import QueryCheckDict

        query_check_dict = QueryCheckDict(settings.prompt.llm_prompt_path)
        response_generator = ResponseGenerator(settings, query_check_dict)

        rag_prompt_template = await response_generator.get_rag_qa_prompt(request.meta.rag_sys_info)

        # 최종 응답 생성을 위한 컨텍스트 준비
        final_prompt_context = {
            "input": request.chat.user,  # 원래 질문 사용
            "rewritten_question": rewritten_question,  # 재작성된 질문도 제공
            "history": formatted_history,  # 형식화된 대화 이력
            "context": retrieval_document,  # 검색된 문서
            "language": language,
            "today": await response_generator.get_today(),
        }

        # VOC 관련 설정 추가 (필요한 경우)
        if request.meta.rag_sys_info == "komico_voc":
            final_prompt_context.update({
                "gw_doc_id_prefix_url": settings.voc.gw_doc_id_prefix_url,
                "check_gw_word_link": settings.voc.check_gw_word_link,
                "check_gw_word": settings.voc.check_gw_word,
                "check_block_line": settings.voc.check_block_line,
            })

        # 최종 응답 생성
        try:
            # 개선된 시스템 프롬프트 생성 (재작성된 질문 포함)
            if "rewritten_question" in final_prompt_context and "{rewritten_question}" not in rag_prompt_template:
                # 템플릿에 재작성된 질문 활용 지시문 추가
                insert_point = rag_prompt_template.find("{input}")
                if insert_point > 0:
                    instruction = "\n\n# 재작성된 질문\n다음은 대화 맥락을 고려하여 명확하게 재작성된 질문입니다. 응답 생성 시 참고하세요:\n{rewritten_question}\n\n# 원래 질문\n"
                    rag_prompt_template = rag_prompt_template[:insert_point] + instruction + rag_prompt_template[
                                                                                             insert_point:]

            # LLM 호출
            answer = await llm_service.ask(
                query=request.chat.user,
                documents=retrieval_document,
                language=language,
                context=final_prompt_context
            )

            if not answer or answer.strip() == "":
                logger.warning(f"[{session_id}] LLM 응답이 비어 있습니다. 대체 메시지를 제공합니다.")
                answer = "죄송합니다. 현재 응답을 생성할 수 없습니다. 잠시 후 다시 시도해 주세요."

            return answer, retrieval_document

        except Exception as e:
            logger.error(f"[{session_id}] 최종 응답 생성 중 오류: {str(e)}")
            return "죄송합니다. 응답 생성 중 오류가 발생했습니다.", retrieval_document

    async def handle_chat_with_history_gemma(self, request: ChatRequest, language: str) -> Tuple[str, List[Document]]:
        """
        Gemma 모델을 위한 이력 기반 채팅 처리

        Args:
            request: 채팅 요청 객체
            language: 응답 언어

        Returns:
            Tuple[str, List[Document]]: (응답 텍스트, 검색된 문서)
        """
        session_id = request.meta.session_id
        logger.debug(f"[{session_id}] Gemma 모델을 위한 이력 처리 시작")

        # 기본적으로 개선된 2단계 접근법과 유사하지만 Gemma 형식 사용
        # 대화 이력 가져오기
        session_history = await self.get_session_history()
        formatted_history = self.format_history_for_gemma(session_history)

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

            # Gemma 형식으로 변환
            rewrite_prompt = self.build_system_prompt_gemma(rewrite_prompt_template, rewrite_context)

            # LLM을 통한 질문 재정의
            from src.common.config_loader import ConfigLoader
            from src.services.core.llm import LLMServiceFactory

            settings = ConfigLoader().get_settings()
            llm_service = await LLMServiceFactory.create_service(settings)

            try:
                # 타임아웃 적용 (최대 3초)
                rewrite_timeout = getattr(settings.llm, 'rewrite_timeout', 3.0)

                rewritten_question = await asyncio.wait_for(
                    llm_service.ask(
                        query=request.chat.user,
                        documents=[Document(page_content=formatted_history)],
                        language=language,
                        context={"task": "rewrite_query", "prompt": rewrite_prompt}
                    ),
                    timeout=rewrite_timeout
                )

                # 재작성된 질문 검증
                if not rewritten_question or len(rewritten_question) < 5:
                    logger.warning(f"[{session_id}] 질문 재정의 실패, 원래 질문 사용")
                    rewritten_question = request.chat.user
                else:
                    logger.debug(f"[{session_id}] 최종 재정의된 질문: '{rewritten_question}'")
            except asyncio.TimeoutError:
                logger.warning(f"[{session_id}] 질문 재정의 타임아웃, 원래 질문 사용")
                rewritten_question = request.chat.user
            except Exception as e:
                logger.error(f"[{session_id}] 질문 재정의 오류: {str(e)}")
                rewritten_question = request.chat.user

        # 검색 수행
        logger.debug(f"[{session_id}] 재정의된 질문으로 문서 검색 시작")
        try:
            # 검색 실행
            retrieval_document = await self.retrieval_source.retrieve(rewritten_question)
            logger.debug(f"[{session_id}] 문서 검색 완료: {len(retrieval_document)}개 문서")
        except Exception as e:
            logger.error(f"[{session_id}] 문서 검색 중 오류: {str(e)}")
            retrieval_document = []

        # 최종 응답 생성
        from src.services.response.generator import ResponseGenerator
        from src.common.query_check_dict import QueryCheckDict

        query_check_dict = QueryCheckDict(settings.prompt.llm_prompt_path)
        response_generator = ResponseGenerator(settings, query_check_dict)

        rag_prompt_template = await response_generator.get_rag_qa_prompt(request.meta.rag_sys_info)

        # 최종 응답 생성을 위한 컨텍스트 준비
        final_prompt_context = {
            "input": request.chat.user,
            "rewritten_question": rewritten_question,
            "history": formatted_history,
            "context": retrieval_document,
            "language": language,
            "today": await response_generator.get_today(),
        }

        # VOC 관련 설정 추가
        if request.meta.rag_sys_info == "komico_voc":
            final_prompt_context.update({
                "gw_doc_id_prefix_url": settings.voc.gw_doc_id_prefix_url,
                "check_gw_word_link": settings.voc.check_gw_word_link,
                "check_gw_word": settings.voc.check_gw_word,
                "check_block_line": settings.voc.check_block_line,
            })

        # Gemma 형식으로 최종 시스템 프롬프트 생성
        vllm_inquery_context = self.build_system_prompt_gemma(rag_prompt_template, final_prompt_context)

        try:
            # LLM 호출
            answer = await llm_service.ask(
                query=request.chat.user,
                documents=retrieval_document,
                language=language,
                context={"prompt": vllm_inquery_context, "is_gemma": True}
            )

            if not answer or answer.strip() == "":
                logger.warning(f"[{session_id}] LLM 응답이 비어 있습니다. 대체 메시지를 제공합니다.")
                answer = "죄송합니다. 현재 응답을 생성할 수 없습니다. 잠시 후 다시 시도해 주세요."

            return answer, retrieval_document

        except Exception as e:
            logger.error(f"[{session_id}] 최종 응답 생성 중 오류: {str(e)}")
            return "죄송합니다. 응답 생성 중 오류가 발생했습니다.", retrieval_document

    async def invoke_chain(self, chain, common_input: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        체인 호출 메서드 (비동기)

        Args:
            chain: 호출할 체인
            common_input: 체인 입력 데이터

        Returns:
            Optional[Dict[str, Any]]: 체인 응답 또는 None
        """
        input_id = (common_input.get("original_question", ""), self.current_session_id)
        if input_id in self.processed_inputs:
            return {"answer": ""}

        try:
            self.processed_inputs.add(input_id)

            if hasattr(chain, "ainvoke") and callable(getattr(chain, "ainvoke")):
                response = await chain.ainvoke(common_input)
            else:
                logger.error(f"[{self.current_session_id}] 제공된 체인에 유효한 'ainvoke' 메서드가 없습니다.")
                return None

            if not response:
                logger.error(f"[{self.current_session_id}] 체인 응답이 비어 있거나 None입니다.")
                return {"answer": "응답을 받을 수 없습니다."}

            return response

        except Exception as e:
            logger.error(f"[{self.current_session_id}] 체인 호출 중 오류: {e}")
            return None
        finally:
            if input_id in self.processed_inputs:
                self.processed_inputs.remove(input_id)

    async def cleanup_processed_sets(self):
        """
        처리된 메시지 ID와 입력 세트를 정리하여 메모리 누수 방지
        """
        # 정기적으로 호출하여 메모리 사용 최적화
        if len(self.processed_message_ids) > 10000:
            logger.warning(f"[{self.current_session_id}] 처리된 메시지 ID가 많습니다. 캐시 정리 진행")
            self.processed_message_ids.clear()

        if len(self.processed_inputs) > 100:
            logger.warning(f"[{self.current_session_id}] 처리된 입력이 많습니다. 캐시 정리 진행")
            self.processed_inputs.clear()


# 클래스 등록
HistoryHandlerFactory.register_handler("standard", StandardHistoryHandler)
