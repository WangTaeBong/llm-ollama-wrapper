"""
기본 히스토리 핸들러 모듈

모든 모델 타입에 공통된 대화 히스토리 관리 기능을 제공합니다.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Set

from dateutil.parser import parse
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from src.common.config_loader import ConfigLoader
from src.schema.chat_req import ChatRequest
from src.schema.retriever_req import RetrieverRequest, RetrieverMeta, RetrieverQuery
from src.schema.vllm_inquery import VllmInquery
from src.services.history.base import HistoryHandlerBase, HistoryStorageBase
from src.services.history.formatters.prompt_formatter import StandardPromptFormatter
from src.services.history.storage.redis_storage import RedisHistoryStorage
from src.services.history.utils.cache_manager import HistoryCacheManager
from src.services.retrieval import create_retriever
from src.services.history.utils.async_helpers import run_with_retry, run_with_timeout

# 설정 로드
settings = ConfigLoader().get_settings()

# 로거 설정
logger = logging.getLogger(__name__)


class BaseHistoryHandler(HistoryHandlerBase):
    """
    기본 히스토리 핸들러 클래스

    모든 모델 타입에 공통된 대화 히스토리 관리 기능을 제공합니다.
    """

    def __init__(
            self,
            llm_model: Any,
            request: ChatRequest,
            max_history_turns: int = 10,
            storage: Optional[HistoryStorageBase] = None,
            cache_manager: Optional[HistoryCacheManager] = None
    ):
        """
        기본 히스토리 핸들러 초기화

        Args:
            llm_model: LLM 모델 인스턴스
            request: 채팅 요청 객체
            max_history_turns: 최대 히스토리 턴 수
            storage: 히스토리 저장소 (기본값: Redis)
            cache_manager: 캐시 관리자 (기본값: 새 인스턴스)
        """
        self.llm_model = llm_model
        self.request = request
        self.current_session_id = request.meta.session_id
        self.current_rag_sys_info = request.meta.rag_sys_info
        self.max_history_turns = max_history_turns

        # 캐시 관리자 초기화
        self.cache_manager = cache_manager or HistoryCacheManager.get_instance()

        # 저장소 초기화
        self.storage = storage or RedisHistoryStorage()

        # 포맷터 초기화
        self.formatter = StandardPromptFormatter()

        # 검색기 초기화
        self.retriever = None

        # 응답 통계
        self.response_stats = None

    async def init_retriever(self, retrieval_documents: List[Document]) -> Any:
        """
        검색기를 초기화합니다.

        Args:
            retrieval_documents: 초기 문서 목록

        Returns:
            Any: 초기화된 검색기
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

        self.retriever = create_retriever(request_data=request_data)

        if retrieval_documents:
            logger.debug(f"[{self.current_session_id}] 검색기에 {len(retrieval_documents)}개 문서 추가")
            self.retriever.add_documents(retrieval_documents)

        return self.retriever

    def get_session_history(self) -> ChatMessageHistory:
        """
        세션 히스토리를 가져옵니다.
        캐시 또는 저장소에서 메시지를 로드하고 시간순으로 정렬합니다.

        Returns:
            ChatMessageHistory: 세션 히스토리
        """
        if not self.current_session_id:
            raise ValueError("Invalid session ID.")

        cache_key = f"{self.current_rag_sys_info}:{self.current_session_id}"

        # 캐시에서 히스토리 확인
        cached_history = self.cache_manager.get("session", cache_key)
        if cached_history:
            return cached_history

        # 새 세션 히스토리 생성
        session_history = ChatMessageHistory()

        try:
            # 저장소에서 메시지 가져오기
            history_messages = self.storage.get_messages(
                self.current_rag_sys_info, self.current_session_id
            )

            # 메시지 처리 및 시간순 정렬
            all_messages = []
            processed_message_ids = set()

            for entry in history_messages:
                for msg in entry.get("messages", []):
                    # 고유 메시지 식별자 생성
                    msg_id = f"{msg['content']}_{msg['timestamp']}"

                    # 중복 메시지 건너뛰기
                    if msg_id in processed_message_ids:
                        continue

                    try:
                        # 타임스탬프 파싱
                        timestamp = parse(msg["timestamp"])
                        all_messages.append((timestamp, msg))
                        processed_message_ids.add(msg_id)
                        # 캐시에 처리된 ID 추가
                        self.cache_manager.add_to_processed_set("message_ids", msg_id)
                    except Exception as e:
                        logger.error(f"[{self.current_session_id}] 타임스탬프 파싱 오류: {e}")

            # 타임스탬프로 메시지 정렬 (오래된 것부터)
            all_messages.sort(key=lambda x: x[0])

            # 최대 히스토리 크기로 제한
            max_messages = self.max_history_turns * 2  # 각 턴은 사용자 메시지와 응답을 포함
            if len(all_messages) > max_messages:
                all_messages = all_messages[-max_messages:]

            # 중복 콘텐츠 제거 (최신 것 유지)
            all_messages = self._remove_duplicate_content_messages(all_messages)

            # 세션 히스토리 초기화
            session_history.clear()

            # 메시지를 세션 히스토리에 추가
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
            self.cache_manager.set("session", cache_key, session_history)

            logger.debug(
                f"[{self.current_session_id}] 저장소에서 {len(session_history.messages)}개 히스토리 메시지 로드됨"
            )

            return session_history

        except Exception as e:
            logger.error(f"[{self.current_session_id}] 세션 히스토리 로드 중 오류: {e}")
            # 오류 시 빈 히스토리 반환
            return session_history

    @classmethod
    def _remove_duplicate_content_messages(
            cls,
            messages: List[Tuple[Any, Dict[str, Any]]]
    ) -> List[Tuple[Any, Dict[str, Any]]]:
        """
        중복 콘텐츠를 가진 메시지 제거 (최신 메시지 유지)

        Args:
            messages: (타임스탬프, 메시지) 튜플 목록

        Returns:
            List[Tuple[Any, Dict[str, Any]]]: 중복이 제거된 메시지 목록
        """
        unique_messages = []
        seen_content = set()

        # 최신 메시지 우선으로 처리하기 위해 역순으로 처리
        for msg_tuple in reversed(messages):
            timestamp, msg = msg_tuple

            # 콘텐츠 비교를 위한 정규화 (공백 제거, 소문자 변환)
            normalized_content = ' '.join(msg["content"].lower().split())

            # 매우 긴 콘텐츠인 경우 앞부분만 사용
            comparison_key = normalized_content[:200] if len(normalized_content) > 200 else normalized_content

            if comparison_key not in seen_content:
                unique_messages.append(msg_tuple)
                seen_content.add(comparison_key)

        # 원래 순서(시간순)로 정렬하여 반환
        unique_messages.sort(key=lambda x: x[0])

        return unique_messages

    async def handle_chat_with_history(
            self,
            request: ChatRequest,
            language: str,
            rag_chat_chain: Any
    ) -> Optional[Dict[str, Any]]:
        """
        히스토리를 활용하여 채팅 요청을 처리합니다.
        이 메서드는 하위 클래스에서 구현해야 합니다.

        Args:
            request: 채팅 요청
            language: 응답 언어
            rag_chat_chain: RAG 채팅 체인

        Returns:
            Optional[Dict[str, Any]]: 채팅 응답 또는 None
        """
        raise NotImplementedError("하위 클래스에서 구현해야 합니다")

    async def handle_chat_with_history_vllm(
            self,
            request: ChatRequest,
            language: str
    ) -> Tuple[str, List[Document]]:
        """
        vLLM을 사용하여 히스토리 기반 채팅 요청을 처리합니다.
        이 메서드는 하위 클래스에서 구현해야 합니다.

        Args:
            request: 채팅 요청
            language: 응답 언어

        Returns:
            Tuple[str, List[Document]]: (응답 텍스트, 검색된 문서 목록)
        """
        raise NotImplementedError("하위 클래스에서 구현해야 합니다")

    async def handle_chat_with_history_vllm_streaming(
            self,
            request: ChatRequest,
            language: str
    ) -> Tuple[VllmInquery, List[Document]]:
        """
        vLLM을 사용하여 스트리밍 모드로 히스토리 기반 채팅 요청을 처리합니다.
        이 메서드는 하위 클래스에서 구현해야 합니다.

        Args:
            request: 채팅 요청
            language: 응답 언어

        Returns:
            Tuple[VllmInquery, List[Document]]: (vLLM 요청 객체, 검색된 문서 목록)
        """
        raise NotImplementedError("하위 클래스에서 구현해야 합니다")

    async def save_chat_history(self, answer: str) -> bool:
        """
        채팅 히스토리를 저장합니다.

        개선된 오류 처리와 재시도 메커니즘을 추가했습니다.

        Args:
            answer: 응답 텍스트

        Returns:
            bool: 저장 성공 여부
        """
        session_id = self.current_session_id
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                # 메시지 데이터 생성
                chat_data = self._create_chat_data(session_id, [
                    self._create_message("HumanMessage", self.request.chat.user),
                    self._create_message("AIMessage", answer)
                ])

                # 저장소에 저장
                success = await self.storage.save_message(
                    system_info=self.current_rag_sys_info,
                    session_id=session_id,
                    message=chat_data
                )

                if success:
                    logger.debug(f"[{session_id}] 대화 이력이 성공적으로 저장됨")
                    return True
                else:
                    # 성공하지 못한 경우 재시도
                    retry_count += 1
                    await asyncio.sleep(0.5 * retry_count)  # 점진적 지연

            except Exception as e:
                retry_count += 1
                logger.warning(
                    f"[{session_id}] 대화 이력 저장 실패 (시도 {retry_count}/{max_retries}): {str(e)}"
                )

                # 마지막 시도가 아니면 잠시 대기 후 재시도
                if retry_count < max_retries:
                    await asyncio.sleep(0.5 * retry_count)  # 점진적 지연

        # 모든 재시도 실패
        logger.error(f"[{session_id}] 모든 대화 이력 저장 시도 실패")
        return False

    @classmethod
    def _create_message(cls, role: str, content: str, timestamp: Optional[str] = None) -> Dict[str, str]:
        """
        메시지 딕셔너리를 생성합니다.

        Args:
            role: 메시지 발신자 역할 (예: "HumanMessage" 또는 "AIMessage")
            content: 메시지 내용
            timestamp: 메시지 타임스탬프 (기본값: 현재 시간)

        Returns:
            Dict[str, str]: 메시지 딕셔너리
        """
        from datetime import datetime

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        return {
            "role": role,
            "content": content,
            "timestamp": timestamp
        }

    @classmethod
    def _create_chat_data(cls, chat_id: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        채팅 데이터 딕셔너리를 생성합니다.

        Args:
            chat_id: 채팅 세션 식별자
            messages: 메시지 딕셔너리 목록

        Returns:
            Dict[str, Any]: 채팅 데이터 딕셔너리
        """
        from datetime import datetime

        return {
            "chat_id": chat_id,
            "messages": [
                {
                    "role": message.get("role"),
                    "content": message.get("content"),
                    "timestamp": message.get("timestamp", datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
                }
                for message in messages
            ]
        }

    async def call_vllm_endpoint(self, data: VllmInquery):
        """
        vLLM 엔드포인트를 호출합니다.

        Args:
            data: vLLM 요청 데이터

        Returns:
            dict: vLLM 응답
        """
        from src.common.restclient import rc

        logger.debug(f"[{self.current_session_id}] vLLM 엔드포인트 호출")
        vllm_url = settings.vllm.endpoint_url

        try:
            # 재시도 메커니즘으로 API 호출
            response = await run_with_retry(
                rc.restapi_post_async(vllm_url, data),
                max_retries=2,
                retry_delay=2,
                session_id=self.current_session_id
            )

            logger.debug(f"[{self.current_session_id}] vLLM 응답 수신 완료")
            return response

        except Exception as e:
            logger.error(f"[{self.current_session_id}] vLLM 엔드포인트 호출 중 오류: {e}")
            # 오류 시 빈 응답 반환
            return {"error": str(e), "generated_text": "", "answer": ""}

    def cleanup_processed_sets(self):
        """
        처리된 세트를 정리하여 메모리 누수를 방지합니다.
        """
        # 캐시 관리자의 정리 메서드 호출
        self.cache_manager.cleanup_processed_sets()
