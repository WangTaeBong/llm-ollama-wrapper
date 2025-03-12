import logging
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
    def format_history_for_prompt(cls, session_history: ChatMessageHistory) -> str:
        """
        Format chat history for prompt in a clean, structured format.

        Args:
            session_history: The chat message history

        Returns:
            Formatted history string
        """
        formatted_history = []

        for msg in session_history.messages:
            # Use roles without timestamps to save tokens
            role = "Human" if isinstance(msg, HumanMessage) else "Assistant"
            formatted_history.append(f"{role}: {msg.content}")

        return "\n".join(formatted_history)

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

    async def handle_chat_with_history_vllm(self,
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
        formatted_history = self.format_history_for_prompt(session_history)
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

    async def handle_chat_with_history_vllm_streaming(self,
                                                      request: ChatRequest,
                                                      language: str):
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
        formatted_history = self.format_history_for_prompt(session_history)
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

        logger.debug(f"[{self.current_session_id}] 스트리밍 요청 준비 완료")

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
