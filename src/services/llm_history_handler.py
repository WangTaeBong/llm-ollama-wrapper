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

from src.common.config_loader import ConfigLoader
from src.common.query_check_dict import QueryCheckDict
from src.common.restclient import rc
from src.schema.chat_req import ChatRequest
from src.schema.retriever_req import RetrieverRequest, RetrieverMeta, RetrieverQuery
from src.schema.vllm_inquery import VllmInquery
from src.services.custom_retriever import CustomRetriever
from src.services.response_generator import ResponseGenerator
from src.utils.prompt import PromptManager
from src.utils.redis_utils import RedisUtils
from src.utils.history_prompt_manager import PromptManager

# Load settings
settings = ConfigLoader().get_settings()

# Configure module-level logger
logger = logging.getLogger(__name__)


class LlmHistoryHandler:
    """
    Handles chat interactions with LLMs by managing session histories and retrieval-based chat chains.
    Optimized for performance, stability, and improved conversation history handling with streaming support.
    """

    session_cache = TTLCache(maxsize=settings.cache.max_concurrent_tasks,
                             ttl=settings.cache.chain_ttl)  # Cache for session histories
    session_chain_cache = TTLCache(maxsize=settings.cache.max_concurrent_tasks,
                                   ttl=settings.cache.chain_ttl)  # Cache for chat chains
    chain_lock = Lock()

    def __init__(self, llm_model: Any, request: ChatRequest, max_history_turns: int = 10):
        """
        Initialize the LlmHistoryHandler.

        Args:
            llm_model (Any): The LLM model to use.
            request (ChatRequest): The chat request containing metadata and user input.
            max_history_turns (int): Maximum number of conversation turns to maintain.
        """
        self.response_stats = None
        self.llm_model = llm_model
        self.current_session_id = request.meta.session_id
        self.current_rag_sys_info = request.meta.rag_sys_info
        self.request = request
        self.max_history_turns = max_history_turns

        self.session_lock = Lock()
        self.processed_inputs = set()
        self.processed_message_ids = set()  # Track message IDs to avoid duplication
        self.retriever: Optional[CustomRetriever] = None

        query_check_dict = QueryCheckDict(settings.prompt.llm_prompt_path)
        self.response_generator = ResponseGenerator(settings, query_check_dict)

    @classmethod
    def is_gemma_model(cls) -> bool:
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
    def build_system_prompt_gemma(cls, system_prompt_template, context):
        """
        Gemma에 맞는 형식으로 시스템 프롬프트를 구성합니다.

        Args:
            system_prompt_template (str): 프롬프트 템플릿
            context (dict): 템플릿에 적용할 변수들

        Returns:
            str: Gemma 형식의 시스템 프롬프트
        """
        try:
            # 먼저 기존 format 메서드로 변수를 대체
            raw_prompt = system_prompt_template.format(**context)

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
            return cls.build_system_prompt_gemma(system_prompt_template, context)
        except Exception as e:
            # 기타 예외 처리
            logger.error(f"Gemma 시스템 프롬프트 형식화 중 오류: {e}")
            # 기본 Gemma 프롬프트로 폴백
            basic_prompt = f"<start_of_turn>user\n다음 질문에 답해주세요: {context.get('input', '질문 없음')}\n<end_of_turn>\n<start_of_turn>model\n"
            return basic_prompt

    async def init_retriever(self, retrieval_documents: List[Document]) -> CustomRetriever:
        """
        Initialize and return a custom retriever.

        Args:
            retrieval_documents (List[Document]): List of documents to initialize the retriever with.

        Returns:
            CustomRetriever: The initialized custom retriever.
        """
        request_data = RetrieverRequest(
            meta=RetrieverMeta(
                company_id=self.request.meta.company_id,
                dept_class=self.request.meta.dept_class,
                rag_sys_info=self.request.meta.rag_sys_info,
                session_id=self.current_session_id,
            ),
            chat=RetrieverQuery(
                user=self.request.chat.user,
                category1=self.request.chat.category1,
                category2=self.request.chat.category2,
                category3=self.request.chat.category3,
            ),
        )

        self.retriever = CustomRetriever(request_data=request_data)

        if retrieval_documents:
            logger.debug(f"[{self.current_session_id}] Adding {len(retrieval_documents)} documents to the retriever.")
            self.retriever.add_documents(retrieval_documents)

        return self.retriever

    def init_chat_chain_with_history(self):
        """
        Initialize or retrieve a cached chat chain with session history.

        Returns:
            Any: The initialized or cached chat chain.
        """
        cache_key = f"{self.current_rag_sys_info}:{self.current_session_id}"

        with self.chain_lock:
            if cache_key in self.session_chain_cache:
                cached_chain = self.session_chain_cache[cache_key]
                if self.is_chain_valid(cached_chain):
                    logger.debug(f"[{self.current_session_id}] Using cached chat chain.")
                    return cached_chain
                logger.warning(f"[{self.current_session_id}] Cached chain is invalid. Recreating chain.")

            rag_prompt_template = self.response_generator.get_rag_qa_prompt(self.current_rag_sys_info)
            rag_chain_prompt = ChatPromptTemplate.from_messages([
                ("system", rag_prompt_template),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ])

            history_aware_retriever = create_history_aware_retriever(
                self.llm_model,
                self.retriever,
                PromptManager.get_contextualize_q_prompt()
            )

            question_answer_chain = create_stuff_documents_chain(
                self.llm_model,
                rag_chain_prompt
            )

            chat_chain = create_retrieval_chain(
                history_aware_retriever,
                question_answer_chain
            )

            if not self.is_chain_valid(chat_chain):
                logger.error(f"[{self.current_session_id}] Newly created chain is invalid.")
                raise ValueError("Chain creation failed.")

            self.session_chain_cache[cache_key] = chat_chain
            logger.debug(f"[{self.current_session_id}] Created and cached new chat chain.")
            return chat_chain

    @staticmethod
    def is_chain_valid(chain) -> bool:
        """
        Validate whether the given chain is callable.

        Args:
            chain (Any): The chat chain to validate.

        Returns:
            bool: True if the chain is valid, False otherwise.
        """
        return hasattr(chain, 'invoke') and callable(getattr(chain, 'invoke'))

    def get_session_history(self) -> ChatMessageHistory:
        """
        Retrieve or initialize session history from cache or Redis.
        Optimized to properly order messages by timestamp and remove duplicates.

        Returns:
            ChatMessageHistory: The session history.
        """
        if not self.current_session_id:
            raise ValueError("Invalid session ID.")

        cache_key = f"{self.current_rag_sys_info}:{self.current_session_id}"

        with self.session_lock:
            # Get cached session history or initialize new one
            session_history = self.session_cache.get(cache_key, ChatMessageHistory())

            try:
                # Fetch messages from Redis
                history_messages = RedisUtils.get_messages_from_redis(
                    self.current_rag_sys_info, self.current_session_id
                )
            except Exception as e:
                logger.error(f"[{self.current_session_id}] Failed to fetch messages from Redis: {e}")
                history_messages = []

            # Process messages with timestamp information
            all_messages = []
            for entry in history_messages:
                for msg in entry.get("messages", []):
                    # Create unique message identifier
                    msg_id = f"{msg['content']}_{msg['timestamp']}"

                    # Skip duplicate messages
                    if msg_id not in self.processed_message_ids:
                        try:
                            timestamp = parse(msg["timestamp"])
                            all_messages.append((timestamp, msg))
                            self.processed_message_ids.add(msg_id)
                        except Exception as e:
                            logger.error(f"[{self.current_session_id}] Timestamp parsing error: {e}")

            # Sort messages by timestamp (ascending - oldest first)
            all_messages.sort(key=lambda x: x[0])

            # Limit history to the most recent conversations
            max_messages = self.max_history_turns * 2  # Each turn consists of a user message and a response
            if len(all_messages) > max_messages:
                logger.debug(f"[{self.current_session_id}] Limiting history to most recent {max_messages} messages")
                all_messages = all_messages[-max_messages:]

            # Remove duplicate content (keeping most recent instances)
            all_messages = self._remove_duplicate_content_messages(all_messages)

            # Clear session history to rebuild with correct order
            session_history.clear()

            # Add messages to session history in correct order
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
                    logger.error(f"[{self.current_session_id}] Message addition error: {e}")

            # Update cache
            self.session_cache[cache_key] = session_history

            logger.debug(
                f"[{self.current_session_id}] Retrieved {len(session_history.messages)} history messages from Redis")

            return session_history

    @classmethod
    def _remove_duplicate_content_messages(
            cls,
            messages: List[Tuple[Any, Dict[str, Any]]]
    ) -> List[Tuple[Any, Dict[str, Any]]]:
        """
        Remove messages with duplicate content, keeping the most recent ones.

        Args:
            messages: List of (timestamp, message) tuples

        Returns:
            List of (timestamp, message) tuples with duplicates removed
        """
        unique_messages = []
        seen_content = set()

        # Process messages in reverse (newest first) to keep most recent duplicate
        for msg_tuple in reversed(messages):
            timestamp, msg = msg_tuple

            # Normalize content for comparison (remove extra spaces, lowercase)
            normalized_content = ' '.join(msg["content"].lower().split())

            # Truncate for comparison if very long
            comparison_key = normalized_content[:200] if len(normalized_content) > 200 else normalized_content

            if comparison_key not in seen_content:
                unique_messages.append(msg_tuple)
                seen_content.add(comparison_key)

        # Re-sort to original order (oldest first)
        unique_messages.sort(key=lambda x: x[0])

        return unique_messages

    @classmethod
    def format_history_for_prompt(cls, session_history: ChatMessageHistory, max_turns: int = 5) -> str:
        """
        형식화된 대화 이력을 생성합니다.
        이전 대화의 맥락을 효과적으로 포착하기 위한 개선된 형식을 사용합니다.

        Args:
            session_history: 채팅 메시지 이력
            max_turns: 포함할 최대 대화 턴 수 (기본값: 5)

        Returns:
            형식화된 대화 이력 문자열
        """
        try:
            # 파라미터 유효성 검사
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
        Gemma 모델에 적합한 형식으로 대화 이력을 구성합니다.
        Gemma의 <start_of_turn>user/<start_of_turn>model 형식을 사용합니다.

        Args:
            session_history: 채팅 메시지 이력
            max_turns: 포함할 최대 대화 턴 수 (기본값: 5)

        Returns:
            str: Gemma 형식의 대화 이력 문자열
        """
        try:
            # 파라미터 유효성 검사
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

    async def handle_chat_with_history(self,
                                       request: ChatRequest,
                                       language: str,
                                       rag_chat_chain: Any) -> Optional[Dict[str, Any]]:
        """
        Handle chat requests using session history.

        Args:
            request (ChatRequest): The chat request.
            language (str): The language of the chat.
            rag_chat_chain (Any): The RAG chat chain to service the request.

        Returns:
            Optional[Dict[str, Any]]: The chat response or None on failure.
        """
        conversational_rag_chain = RunnableWithMessageHistory(
            runnable=rag_chat_chain,
            get_session_history=self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_message_key="answer",
        )

        common_input = {
            "original_question": request.chat.user,
            "input": request.chat.user,
            "history": "",  # Empty for ollama processing
            "language": language,
            "today": self.response_generator.get_today(),
        }

        if self.request.meta.rag_sys_info == "komico_voc":
            common_input.update({
                "gw_doc_id_prefix_url": settings.voc.gw_doc_id_prefix_url,
                "check_gw_word_link": settings.voc.check_gw_word_link,
                "check_gw_word": settings.voc.check_gw_word,
                "check_block_line": settings.voc.check_block_line,
            })

        try:
            logger.debug(f"[{self.current_session_id}] Invoking RAG chain with history")
            response = await self.invoke_chain(conversational_rag_chain, common_input)
            logger.debug(f"[{self.current_session_id}] RAG chain invocation completed successfully")
        except Exception as e:
            logger.error(f"[{self.current_session_id}] Unexpected error occurred: {e}")
            return None

        if not response or "answer" not in response:
            logger.error(f"[{self.current_session_id}] Invalid response format: {response}")
            return None

        return response

    async def handle_chat_with_history_vllm(self, request: ChatRequest, language: str):
        """
        모델 유형에 따라 적절한 히스토리 핸들러로 디스패치합니다.

        Args:
            request (ChatRequest): 채팅 요청
            language (str): 언어 코드

        Returns:
            tuple: (answer, retrieval_document)
        """
        if self.__class__.is_gemma_model():
            logger.info(f"[{self.current_session_id}] Gemma 모델 감지됨, Gemma 전용 핸들러로 처리")
            return await self.handle_chat_with_history_gemma(request, language)
        else:
            # 기존 개선된 핸들러 사용
            if getattr(settings.llm, 'use_improved_history', False):
                return await self.handle_chat_with_history_vllm_improved(request, language)
            else:
                # 기존 방식 사용
                return await self._handle_chat_with_history_vllm_original(request, language)

    async def _handle_chat_with_history_vllm_original(self,
                                                      request: ChatRequest,
                                                      language: str):
        """
        Handle chat requests using session history with VLLM.
        Optimized to use formatted history and avoid redundancy.

        Args:
            request (ChatRequest): The chat request.
            language (str): The language of the chat.

        Returns:
            tuple: (answer, retrieval_document)
        """
        # Get prompt template
        rag_prompt_template = self.response_generator.get_rag_qa_prompt(self.current_rag_sys_info)

        # Get retrieval documents
        logger.debug(f"[{self.current_session_id}] Retrieving documents...")
        retrieval_document = await self.retriever.ainvoke(request.chat.user)
        logger.debug(f"[{self.current_session_id}] Retrieved {len(retrieval_document)} documents")

        # Get chat history in structured format
        session_history = self.get_session_history()
        formatted_history = self.format_history_for_prompt(session_history, settings.llm.max_history_turns)
        logger.debug(f"[{self.current_session_id}] Processed {len(session_history.messages)} history messages for VLLM")

        # Prepare context for prompt
        common_input = {
            "input": request.chat.user,
            "history": formatted_history,  # Using optimized format
            "context": retrieval_document,
            "language": language,
            "today": self.response_generator.get_today(),
        }

        if self.request.meta.rag_sys_info == "komico_voc":
            common_input.update({
                "gw_doc_id_prefix_url": settings.voc.gw_doc_id_prefix_url,
                "check_gw_word_link": settings.voc.check_gw_word_link,
                "check_gw_word": settings.voc.check_gw_word,
                "check_block_line": settings.voc.check_block_line,
            })

        # Build prompt and create request
        vllm_inquery_context = self.build_system_prompt(rag_prompt_template, common_input)
        vllm_request = VllmInquery(
            request_id=self.request.meta.session_id,
            prompt=vllm_inquery_context
        )

        # Call VLLM endpoint
        logger.debug(f"[{self.current_session_id}] Calling VLLM endpoint")
        response = await self.call_vllm_endpoint(vllm_request)
        answer = response.get("generated_text", "") or response.get("answer", "")
        logger.debug(f"[{self.current_session_id}] VLLM response received, length: {len(answer)}")

        return answer, retrieval_document

    async def handle_chat_with_history_vllm_improved(self,
                                                     request: ChatRequest,
                                                     language: str):
        """
        vLLM을 사용하여 2단계 접근법으로 대화 이력을 처리합니다.
        1. 대화 이력과 현재 질문을 사용하여 독립적인 질문 생성
        2. 독립적인 질문으로 문서를 검색하고 최종 응답 생성

        Args:
            request (ChatRequest): 채팅 요청
            language (str): 언어 코드

        Returns:
            tuple: (answer, retrieval_document)
        """
        logger.debug(f"[{self.current_session_id}] 개선된 vLLM 히스토리 처리 시작")

        # 1단계: 대화 이력을 사용하여 독립적인 질문 생성
        # 대화 이력 가져오기
        session_history = self.get_session_history()
        formatted_history = self.format_history_for_prompt(session_history, settings.llm.max_history_turns)

        # 대화 이력이 없는 경우 바로 원래 질문 사용
        if not formatted_history or len(session_history.messages) == 0:
            logger.debug(f"[{self.current_session_id}] 대화 이력이 없어 원래 질문을 사용합니다.")
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

            rewrite_prompt = rewrite_prompt_template.format(**rewrite_context)

            # vLLM에 질문 재정의 요청
            rewrite_request = VllmInquery(
                request_id=f"{self.current_session_id}_rewrite",
                prompt=rewrite_prompt
            )

            logger.debug(f"[{self.current_session_id}] 질문 재정의 vLLM 요청 전송")
            rewrite_response = await self.call_vllm_endpoint(rewrite_request)
            rewritten_question = rewrite_response.get("generated_text", "").strip()

            # 재작성된 질문이 없거나 오류 발생 시 원래 질문 사용
            if not rewritten_question or len(rewritten_question) < 5:
                logger.warning(f"[{self.current_session_id}] 질문 재정의 실패, 원래 질문 사용")
                rewritten_question = request.chat.user
            else:
                logger.debug(f"[{self.current_session_id}] 최종 재정의 사용 질문: '{rewritten_question}'")
                """
                # 질문 재정의 후 검증 추가
                if rewritten_question:
                    # 원본 질문에서 중요 엔티티 추출
                    important_entities = self._extract_important_entities(request.chat.user)

                    # 재작성된 질문 검증
                    rewritten_question = self.validate_rewritten_question(
                        request.chat.user,
                        rewritten_question,
                        important_entities
                    )

                    logger.debug(f"[{self.current_session_id}] 최종 재정의 사용 질문: '{rewritten_question}'")
                """

        # 2단계: 재정의된 질문으로 문서 검색
        logger.debug(f"[{self.current_session_id}] 재정의된 질문으로 문서 검색 시작")
        retrieval_document = await self.retriever.ainvoke(rewritten_question)
        logger.debug(f"[{self.current_session_id}] 문서 검색 완료: {len(retrieval_document)}개 문서")

        # 3단계: 최종 응답 생성
        # RAG 프롬프트 템플릿 가져오기
        rag_prompt_template = self.response_generator.get_rag_qa_prompt(self.current_rag_sys_info)

        # 최종 응답 생성을 위한 컨텍스트 준비
        final_prompt_context = {
            "input": request.chat.user,  # 원래 질문 사용
            "rewritten_question": rewritten_question,  # 재작성된 질문도 제공
            "history": formatted_history,  # 형식화된 대화 이력
            "context": retrieval_document,  # 검색된 문서
            "language": language,
            "today": self.response_generator.get_today(),
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
        vllm_inquery_context = self.build_system_prompt_improved(rag_prompt_template, final_prompt_context)

        # vLLM에 최종 응답 요청
        vllm_request = VllmInquery(
            request_id=self.request.meta.session_id,
            prompt=vllm_inquery_context
        )

        logger.debug(f"[{self.current_session_id}] 최종 응답 생성 vLLM 요청 전송")
        response = await self.call_vllm_endpoint(vllm_request)
        answer = response.get("generated_text", "") or response.get("answer", "")
        logger.debug(f"[{self.current_session_id}] 최종 응답 생성 완료, 응답 길이: {len(answer)}")

        return answer, retrieval_document

    async def handle_chat_with_history_vllm_streaming(self, request: ChatRequest, language: str):
        """
        모델 유형에 따라 적절한 스트리밍 히스토리 핸들러로 디스패치합니다.

        Args:
            request (ChatRequest): 채팅 요청
            language (str): 언어 코드

        Returns:
            tuple: (vllm_request, retrieval_document)
        """
        if self.__class__.is_gemma_model():
            logger.info(f"[{self.current_session_id}] Gemma 모델 감지됨, Gemma 스트리밍 핸들러로 처리")
            return await self.handle_chat_with_history_gemma_streaming(request, language)
        else:
            # 기존 개선된 핸들러 사용
            if getattr(settings.llm, 'use_improved_history', False):
                return await self.handle_chat_with_history_vllm_streaming_improved(request, language)
            else:
                # 기존 방식 사용
                return await self._handle_chat_with_history_vllm_streaming_original(request, language)

    async def _handle_chat_with_history_vllm_streaming_original(self, request: ChatRequest, language: str):
        """
        VLLM을 사용하여 스트리밍 모드로 세션 히스토리 기반 채팅 요청 처리.

        최적화된 대화 히스토리 관리와 중복 제거를 통해 효율적인 스트리밍을 지원합니다.

        Args:
            request (ChatRequest): 채팅 요청.
            language (str): 채팅 언어.

        Returns:
            tuple: (vllm_request, retrieval_document) - 스트리밍용 요청 객체와 검색된 문서
        """
        # 프롬프트 템플릿 가져오기
        rag_prompt_template = self.response_generator.get_rag_qa_prompt(self.current_rag_sys_info)

        # 검색 문서 가져오기
        logger.debug(f"[{self.current_session_id}] 스트리밍용 검색 문서 가져오기 시작.")
        retrieval_document = await self.retriever.ainvoke(request.chat.user)
        logger.debug(f"[{self.current_session_id}] 스트리밍용 검색 문서 가져오기 완료: {len(retrieval_document)}개")

        # 채팅 히스토리 가져오기 및 최적화
        session_history = self.get_session_history()
        formatted_history = self.format_history_for_prompt(session_history, settings.llm.max_history_turns)
        logger.debug(f"[{self.current_session_id}] 스트리밍용 {len(session_history.messages)}개 히스토리 메시지 처리 완료")

        # 컨텍스트 준비
        common_input = {
            "input": request.chat.user,
            "history": formatted_history,  # 최적화된 형식 사용
            "context": retrieval_document,
            "language": language,
            "today": self.response_generator.get_today(),
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
        vllm_inquery_context = self.build_system_prompt(rag_prompt_template, common_input)

        # 스트리밍을 위한 vLLM 요청 생성
        vllm_request = VllmInquery(
            request_id=self.current_session_id,
            prompt=vllm_inquery_context,
            stream=True  # 스트리밍 모드 활성화
        )

        logger.info(f"[{self.current_session_id}] 스트리밍 요청을 {settings.vllm.endpoint_url}로 전송합니다")
        logger.debug(f"[{self.current_session_id}] 스트리밍 요청 준비 완료")

        return vllm_request, retrieval_document

    async def handle_chat_with_history_vllm_streaming_improved(self,
                                                               request: ChatRequest,
                                                               language: str):
        """
        vLLM을 사용하여 2단계 접근법으로 스트리밍 모드의 대화 이력을 처리합니다.
        1. 대화 이력과 현재 질문을 사용하여 독립적인 질문 생성
        2. 독립적인 질문으로 문서를 검색하고 스트리밍 응답 생성

        Args:
            request (ChatRequest): 채팅 요청
            language (str): 언어 코드

        Returns:
            tuple: (vllm_request, retrieval_document) - 스트리밍용 요청 객체와 검색된 문서
        """
        session_id = self.current_session_id
        logger.debug(f"[{session_id}] 개선된 vLLM 스트리밍 히스토리 처리 시작")

        # 1단계: 대화 이력을 사용하여 독립적인 질문 생성
        # 대화 이력 가져오기
        session_history = self.get_session_history()
        formatted_history = self.format_history_for_prompt(session_history, settings.llm.max_history_turns)

        # 대화 이력이 없는 경우 바로 원래 질문 사용
        if not formatted_history or len(session_history.messages) == 0:
            logger.debug(f"[{session_id}] 대화 이력이 없어 원래 질문을 사용합니다.")
            rewritten_question = request.chat.user
        else:
            # 질문 재정의를 위한 프롬프트 템플릿
            # JSON 파일에서 템플릿 불러오기
            from src.utils.prompt import PromptManager
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
                    logger.debug(f"[{session_id}] 최종 재정의 사용 질문: '{rewritten_question}'")
                    """
                    # 질문 재정의 후 검증 추가
                    if rewritten_question:
                        # 원본 질문에서 중요 엔티티 추출
                        important_entities = self._extract_important_entities(request.chat.user)

                        # 재작성된 질문 검증
                        rewritten_question = self.validate_rewritten_question(
                            request.chat.user,
                            rewritten_question,
                            important_entities
                        )
                    """
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
        rag_prompt_template = self.response_generator.get_rag_qa_prompt(self.current_rag_sys_info)

        # 최종 응답 생성을 위한 컨텍스트 준비
        final_prompt_context = {
            "input": request.chat.user,  # 원래 질문 사용
            "rewritten_question": rewritten_question,  # 재작성된 질문도 제공
            "history": formatted_history,  # 형식화된 대화 이력
            "context": retrieval_document,  # 검색된 문서
            "language": language,
            "today": self.response_generator.get_today(),
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
        try:
            # 일단 build_system_prompt_improved 함수가 있는지 확인
            if hasattr(self, 'build_system_prompt_improved') and callable(
                    getattr(self, 'build_system_prompt_improved')):
                vllm_inquery_context = self.build_system_prompt_improved(rag_prompt_template, final_prompt_context)
            else:
                # 없으면 기존 함수 사용
                import re
                # build_system_prompt_improved를 인라인으로 구현
                # 템플릿에 재작성된 질문 주입을 위한 토큰 추가
                if "rewritten_question" in final_prompt_context and "{rewritten_question}" not in rag_prompt_template:
                    # 템플릿에 재작성된 질문 활용 지시문 추가
                    insert_point = rag_prompt_template.find("{input}")
                    if insert_point > 0:
                        instruction = "\n\n# 재작성된 질문\n다음은 대화 맥락을 고려하여 명확하게 재작성된 질문입니다. 응답 생성 시 참고하세요:\n{rewritten_question}\n\n# 원래 질문\n"
                        rag_prompt_template = rag_prompt_template[:insert_point] + instruction + rag_prompt_template[
                                                                                                 insert_point:]

                # 모든 필수 키가 있는지 확인
                required_keys = set()
                for match in re.finditer(r"{(\w+)}", rag_prompt_template):
                    required_keys.add(match.group(1))

                # 누락된 키가 있으면 빈 문자열로 대체
                for key in required_keys:
                    if key not in final_prompt_context:
                        logger.warning(f"시스템 프롬프트 템플릿에 필요한 키가 누락됨: {key}, 빈 문자열로 대체합니다.")
                        final_prompt_context[key] = ""

                # 템플릿 형식화
                vllm_inquery_context = rag_prompt_template.format(**final_prompt_context)
        except Exception as e:
            logger.error(f"[{session_id}] 프롬프트 빌드 오류: {str(e)}")
            # 오류 발생 시 기본 방식으로 폴백
            vllm_inquery_context = self.build_system_prompt(rag_prompt_template, final_prompt_context)

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

        logger.debug(f"[{session_id}] 스트리밍 요청 준비 완료")

        # 성능 개선을 위한 통계 측정
        self.response_stats = {
            "rewrite_time": 0,
            "retrieval_time": 0,
            "total_prep_time": time.time()
        }

        return vllm_request, retrieval_document

    async def handle_chat_with_history_gemma(self,
                                             request: ChatRequest,
                                             language: str):
        """
        Gemma 모델을 사용하여 2단계 접근법으로 대화 이력을 처리합니다.
        1. 대화 이력과 현재 질문을 사용하여 독립적인 질문 생성
        2. 독립적인 질문으로 문서를 검색하고 최종 응답 생성

        Args:
            request (ChatRequest): 채팅 요청
            language (str): 언어 코드

        Returns:
            tuple: (answer, retrieval_document)
        """
        logger.debug(f"[{self.current_session_id}] Gemma 모델을 위한 히스토리 처리 시작")

        # 1단계: 대화 이력을 사용하여 독립적인 질문 생성
        # 대화 이력 가져오기
        session_history = self.get_session_history()
        formatted_history = self.format_history_for_gemma(session_history, settings.llm.max_history_turns)

        # 대화 이력이 없는 경우 바로 원래 질문 사용
        if not formatted_history or len(session_history.messages) == 0:
            logger.debug(f"[{self.current_session_id}] 대화 이력이 없어 원래 질문을 사용합니다.")
            rewritten_question = request.chat.user
        else:
            # 질문 재정의를 위한 프롬프트 템플릿 (Gemma 형식으로 직접 구성)
            # JSON 파일에서 템플릿 불러오기
            rewrite_prompt_template = PromptManager.get_rewrite_prompt_template()

            # 질문 재정의 프롬프트 생성
            rewrite_context = {
                "history": formatted_history,
                "input": request.chat.user,
            }

            # Gemma 형식으로 프롬프트 변환
            rewrite_prompt = self.build_system_prompt_gemma(rewrite_prompt_template, rewrite_context)

            # vLLM에 질문 재정의 요청
            rewrite_request = VllmInquery(
                request_id=f"{self.current_session_id}_rewrite",
                prompt=rewrite_prompt
            )

            logger.debug(f"[{self.current_session_id}] 질문 재정의 Gemma 요청 전송")
            rewrite_response = await self.call_vllm_endpoint(rewrite_request)
            rewritten_question = rewrite_response.get("generated_text", "").strip()

            # 재작성된 질문이 없거나 오류 발생 시 원래 질문 사용
            if not rewritten_question or len(rewritten_question) < 5:
                logger.warning(f"[{self.current_session_id}] 질문 재정의 실패, 원래 질문 사용")
                rewritten_question = request.chat.user
            else:
                logger.debug(f"[{self.current_session_id}] 질문 재정의 성공: '{rewritten_question}'")

        # 2단계: 재정의된 질문으로 문서 검색
        logger.debug(f"[{self.current_session_id}] 재정의된 질문으로 문서 검색 시작")
        retrieval_document = await self.retriever.ainvoke(rewritten_question)
        logger.debug(f"[{self.current_session_id}] 문서 검색 완료: {len(retrieval_document)}개 문서")

        # 3단계: 최종 응답 생성
        # RAG 프롬프트 템플릿 가져오기
        rag_prompt_template = self.response_generator.get_rag_qa_prompt(self.current_rag_sys_info)

        # 최종 응답 생성을 위한 컨텍스트 준비
        final_prompt_context = {
            "input": request.chat.user,  # 원래 질문 사용
            "rewritten_question": rewritten_question,  # 재작성된 질문도 제공
            "history": formatted_history,  # 형식화된 대화 이력
            "context": retrieval_document,  # 검색된 문서
            "language": language,
            "today": self.response_generator.get_today(),
        }

        # VOC 관련 설정 추가 (필요한 경우)
        if self.request.meta.rag_sys_info == "komico_voc":
            final_prompt_context.update({
                "gw_doc_id_prefix_url": settings.voc.gw_doc_id_prefix_url,
                "check_gw_word_link": settings.voc.check_gw_word_link,
                "check_gw_word": settings.voc.check_gw_word,
                "check_block_line": settings.voc.check_block_line,
            })

        # Gemma 형식으로 최종 시스템 프롬프트 생성
        vllm_inquery_context = self.build_system_prompt_gemma(rag_prompt_template, final_prompt_context)

        # vLLM에 최종 응답 요청
        vllm_request = VllmInquery(
            request_id=self.request.meta.session_id,
            prompt=vllm_inquery_context
        )

        logger.debug(f"[{self.current_session_id}] 최종 응답 생성 Gemma 요청 전송")
        response = await self.call_vllm_endpoint(vllm_request)
        answer = response.get("generated_text", "") or response.get("answer", "")
        logger.debug(f"[{self.current_session_id}] 최종 응답 생성 완료, 응답 길이: {len(answer)}")

        return answer, retrieval_document

    async def handle_chat_with_history_gemma_streaming(self,
                                                       request: ChatRequest,
                                                       language: str):
        """
        Gemma 모델을 사용하여 2단계 접근법으로 스트리밍 모드의 대화 이력을 처리합니다.
        1. 대화 이력과 현재 질문을 사용하여 독립적인 질문 생성
        2. 독립적인 질문으로 문서를 검색하고 스트리밍 응답 생성

        Args:
            request (ChatRequest): 채팅 요청
            language (str): 언어 코드

        Returns:
            tuple: (vllm_request, retrieval_document) - 스트리밍용 요청 객체와 검색된 문서
        """
        session_id = self.current_session_id
        logger.debug(f"[{session_id}] Gemma 모델을 위한 스트리밍 히스토리 처리 시작")

        # 1단계: 대화 이력을 사용하여 독립적인 질문 생성
        # 대화 이력 가져오기
        session_history = self.get_session_history()
        formatted_history = self.format_history_for_gemma(session_history, settings.llm.max_history_turns)

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

            # Gemma 형식으로 프롬프트 변환
            rewrite_prompt = self.build_system_prompt_gemma(rewrite_prompt_template, rewrite_context)

            # vLLM에 질문 재정의 요청
            rewrite_request = VllmInquery(
                request_id=f"{session_id}_rewrite",
                prompt=rewrite_prompt
            )

            logger.debug(f"[{session_id}] 질문 재정의 Gemma 요청 전송")
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
                    # 질문 재정의 후 검증 추가
                    if rewritten_question:
                        logger.debug(f"[{session_id}] 최종 재정의 사용 질문: '{rewritten_question}'")
                        """
                        # 원본 질문에서 중요 엔티티 추출
                        important_entities = self._extract_important_entities(request.chat.user)

                        # 재작성된 질문 검증
                        rewritten_question = self.validate_rewritten_question(
                            request.chat.user,
                            rewritten_question,
                            important_entities
                        )
                    """
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
        rag_prompt_template = self.response_generator.get_rag_qa_prompt(self.current_rag_sys_info)

        # 최종 응답 생성을 위한 컨텍스트 준비
        final_prompt_context = {
            "input": request.chat.user,  # 원래 질문 사용
            "rewritten_question": rewritten_question,  # 재작성된 질문도 제공
            "history": formatted_history,  # 형식화된 대화 이력
            "context": retrieval_document,  # 검색된 문서
            "language": language,
            "today": self.response_generator.get_today(),
        }

        # VOC 관련 설정 추가 (필요한 경우)
        if request.meta.rag_sys_info == "komico_voc":
            final_prompt_context.update({
                "gw_doc_id_prefix_url": settings.voc.gw_doc_id_prefix_url,
                "check_gw_word_link": settings.voc.check_gw_word_link,
                "check_gw_word": settings.voc.check_gw_word,
                "check_block_line": settings.voc.check_block_line,
            })

        # 최종 시스템 프롬프트 생성 (Gemma 형식)
        vllm_inquery_context = self.build_system_prompt_gemma(rag_prompt_template, final_prompt_context)
        # logger.debug(f"[{session_id}] Gemma prompt: {vllm_inquery_context}")

        # 스트리밍을 위한 vLLM 요청 생성
        vllm_request = VllmInquery(
            request_id=session_id,
            prompt=vllm_inquery_context,
            stream=True  # 스트리밍 모드 활성화
        )

        logger.debug(f"[{session_id}] Gemma 스트리밍 요청 준비 완료")

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

    async def call_vllm_endpoint(self, data: VllmInquery):
        """
        Call the VLLM endpoint with the provided data.

        Args:
            data (VllmInquery): The request data for VLLM.

        Returns:
            dict: The response from VLLM.
        """
        logger.debug(f"[{self.current_session_id}] Calling VLLM endpoint.")
        vllm_url = settings.vllm.endpoint_url

        try:
            response = await rc.restapi_post_async(vllm_url, data)
            logger.debug(f"[{self.current_session_id}] VLLM endpoint response received successfully.")
            return response
        except Exception as e:
            logger.error(f"[{self.current_session_id}] Error calling VLLM endpoint: {e}")
            # Return empty response instead of raising to maintain stability
            return {"error": str(e), "generated_text": "", "answer": ""}

    @classmethod
    def build_system_prompt(cls, system_prompt_template, context):
        """
        Build the system prompt by applying dynamic variables to the template.
        Enhanced with error handling for missing keys.

        Args:
            system_prompt_template (str): The prompt template.
            context (dict): Variables to be applied to the template.

        Returns:
            str: The formatted system prompt.
        """
        try:
            return system_prompt_template.format(**context)
        except KeyError as e:
            # Handle missing keys gracefully
            missing_key = str(e).strip("'")
            logger.warning(f"Missing key in system prompt template: {missing_key}, setting to empty string")
            context[missing_key] = ""
            return system_prompt_template.format(**context)
        except Exception as e:
            logger.error(f"Error formatting system prompt: {e}")
            # Fallback to basic prompt if formatting fails completely
            return f"Answer the following question based on the context: {context.get('input', 'No input provided')}"

    @classmethod
    def build_system_prompt_improved(cls, system_prompt_template, context):
        """
        개선된 시스템 프롬프트 빌드 메소드.
        재작성된 질문을 포함하고 오류 처리를 강화했습니다.

        Args:
            system_prompt_template (str): 프롬프트 템플릿
            context (dict): 템플릿에 적용할 변수들

        Returns:
            str: 형식화된 시스템 프롬프트
        """
        try:
            # 템플릿에 재작성된 질문 주입을 위한 토큰 추가
            if "rewritten_question" in context and "{rewritten_question}" not in system_prompt_template:
                # 템플릿에 재작성된 질문 활용 지시문 추가
                insert_point = system_prompt_template.find("{input}")
                if insert_point > 0:
                    instruction = "\n\n# 재작성된 질문\n다음은 대화 맥락을 고려하여 명확하게 재작성된 질문입니다. 응답 생성 시 참고하세요:\n{rewritten_question}\n\n# 원래 질문\n"
                    system_prompt_template = system_prompt_template[
                                             :insert_point] + instruction + system_prompt_template[insert_point:]

            # 모든 필수 키가 있는지 확인
            required_keys = set()
            for match in re.finditer(r"{(\w+)}", system_prompt_template):
                required_keys.add(match.group(1))

            # 누락된 키가 있으면 빈 문자열로 대체
            for key in required_keys:
                if key not in context:
                    logger.warning(f"시스템 프롬프트 템플릿에 필요한 키가 누락됨: {key}, 빈 문자열로 대체합니다.")
                    context[key] = ""

            # 템플릿 형식화
            return system_prompt_template.format(**context)

        except KeyError as e:
            # 누락된 키 처리
            missing_key = str(e).strip("'")
            logger.warning(f"시스템 프롬프트 템플릿에 키가 누락됨: {missing_key}, 빈 문자열로 대체합니다.")
            context[missing_key] = ""
            return system_prompt_template.format(**context)
        except Exception as e:
            # 기타 예외 처리
            logger.error(f"시스템 프롬프트 형식화 중 오류: {e}")
            # 기본 프롬프트로 폴백
            return f"다음 컨텍스트를 기반으로 질문에 답하세요:\n\n컨텍스트: {context.get('context', '')}\n\n질문: {context.get('input', '질문 없음')}"

    async def invoke_chain(self, rag_chain, common_input: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Execute the RAG chain, handling asynchronous methods.

        Args:
            rag_chain (Any): The RAG chain to invoke.
            common_input (Dict[str, Any]): The input data for the chain.

        Returns:
            Optional[Dict[str, Any]]: The response from the RAG chain or None.
        """
        input_id = (common_input["original_question"], self.current_session_id)
        if input_id in self.processed_inputs:
            return {"answer": ""}

        try:
            self.processed_inputs.add(input_id)

            if hasattr(rag_chain, "ainvoke") and callable(getattr(rag_chain, "ainvoke")):
                response = await rag_chain.ainvoke(common_input)
            else:
                logger.error(f"[{self.current_session_id}] The provided chain does not have a valid 'ainvoke' method.")
                return None

            if not response:
                logger.error(f"[{self.current_session_id}] Chain response is empty or None.")
                return {"answer": "No response from the model."}

            return response

        except Exception as e:
            logger.error(f"[{self.current_session_id}] Error during chain invocation: {e}")
            return None
        finally:
            if input_id in self.processed_inputs:
                self.processed_inputs.remove(input_id)

    async def update_chat_chain(self):
        """
        Reinitialize the chat chain for the current session.

        Returns:
            Any: The new chat chain.
        """
        cache_key = f"{self.current_rag_sys_info}:{self.current_session_id}"
        self.session_chain_cache.pop(cache_key, None)
        logger.debug(f"[{self.current_session_id}] Chat chain cache cleared, recreating chain")
        return self.init_chat_chain_with_history()

    def cleanup_processed_sets(self):
        """
        정기적으로 처리된 메시지 ID와 입력 세트를 정리하여 메모리 누수 방지
        """
        # 이 메서드는 주기적으로 호출되어 메모리 사용을 최적화
        if len(self.processed_message_ids) > 10000:
            logger.warning(f"[{self.current_session_id}] 처리된 메시지 ID가 많습니다. 캐시 정리 진행")
            self.processed_message_ids.clear()

        if len(self.processed_inputs) > 100:
            logger.warning(f"[{self.current_session_id}] 처리된 입력이 많습니다. 캐시 정리 진행")
            self.processed_inputs.clear()

    def validate_rewritten_question(self, original_question, rewritten_question, important_entities=None):
        """
        재작성된 질문의 품질을 검증하고 적합하지 않을 경우 원본 질문을 반환합니다.

        Args:
            original_question (str): 원본 질문
            rewritten_question (str): 재작성된 질문
            important_entities (list, optional): 보존해야 할 중요 엔티티 목록

        Returns:
            str: 검증을 통과한 재작성 질문 또는 원본 질문
        """
        # 중요 엔티티가 지정되지 않은 경우, 원본 질문에서 추출
        if important_entities is None:
            important_entities = self._extract_important_entities(original_question)

        # 1. 핵심 엔티티 보존 검증
        entities_preserved = all(entity in rewritten_question for entity in important_entities)

        # 2. 의미적 일관성 검증 (간소화된 버전)
        # 정교한 임베딩 비교는 추가 라이브러리가 필요하므로, 간단한 단어 중복 기반 유사도 사용
        original_words = set(original_question.lower().split())
        rewritten_words = set(rewritten_question.lower().split())
        word_overlap = len(original_words.intersection(rewritten_words)) / max(len(original_words), 1)

        # 3. 길이 및 복잡성 검증
        length_ratio = len(rewritten_question) / max(len(original_question), 1)
        length_appropriate = 0.7 <= length_ratio <= 2.5

        # 로깅
        session_id = getattr(self, 'current_session_id', 'unknown')
        logger.debug(f"[{session_id}] 질문 재작성 검증: 엔티티 보존={entities_preserved}, "
                     f"단어 유사도={word_overlap:.2f}, 길이 비율={length_ratio:.2f}")

        # 판단 로직
        if not entities_preserved:
            logger.warning(f"[{session_id}] 엔티티 보존 실패로 원본 질문 사용")
            return original_question

        if word_overlap < 0.4:  # 단어 기반 유사도 임계값
            logger.warning(f"[{session_id}] 의미 유사도 낮음으로 원본 질문 사용")
            return original_question

        if not length_appropriate:
            logger.warning(f"[{session_id}] 길이 비율 부적절로 원본 질문 사용")
            return original_question

        logger.debug(f"[{session_id}] 재작성된 질문 검증 통과")
        return rewritten_question

    @classmethod
    def _extract_important_entities(cls, text):
        """
        텍스트에서 중요 엔티티(고유명사, 전문용어 등)를 추출합니다.
        간단한 구현으로, 대문자로 시작하는 단어와 숫자를 포함하는 단어를 추출합니다.

        Args:
            text (str): 분석할 텍스트

        Returns:
            list: 추출된 중요 엔티티 목록
        """
        # 한글 고유명사 추출 패턴 (2글자 이상 연속된 한글)
        korean_entities = re.findall(r'[가-힣]{2,}', text)

        # 영문 대문자로 시작하는 단어 추출
        capitalized_words = re.findall(r'\b[A-Z][a-zA-Z]*\b', text)

        # 숫자가 포함된 단어 추출
        numeric_entities = re.findall(r'\b\w*\d+\w*\b', text)

        # 결과 결합 및 중복 제거
        entities = list(set(korean_entities + capitalized_words + numeric_entities))

        # 불용어 제거 (필요시 확장)
        stopwords = ['그것', '이것', '저것', '그', '이', '저', '그런', '이런', '저런']
        entities = [e for e in entities if e not in stopwords]

        return entities
