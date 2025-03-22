"""
문서 유효성 검증 모듈

문서의 유효성을 검증하고 오류 응답을 생성하는 기능을 제공합니다.
"""

import logging
from typing import Dict, Optional, Any

from src.common.error_cd import ErrorCd
from src.schema.chat_req import ChatRequest
from src.schema.chat_res import ChatResponse, MetaRes, ChatRes
from src.services.document_processor.base import DocumentValidatorBase

# 로거 설정
logger = logging.getLogger(__name__)


class DocumentValidator(DocumentValidatorBase):
    """
    문서 유효성 검증 클래스

    검색 문서의 유효성을 검증하고 필요한 경우 적절한 응답을 생성합니다.
    """

    def __init__(self, settings: Any):
        """
        문서 유효성 검증기 초기화

        Args:
            settings: 설정 객체
        """
        self.settings = settings
        self._system_messages = self._initialize_system_messages()

    @classmethod
    def _initialize_system_messages(cls) -> Dict[str, str]:
        """
        다국어 시스템 메시지를 초기화합니다.

        Returns:
            Dict[str, str]: 언어별 시스템 메시지 사전
        """
        return {
            "en": "We're unable to generate an answer based on our knowledge base for your question. "
                  "Please try to be more specific with your question.",
            "jp": "質問された内容について、ナレッジベースで回答を生成することができません。 質問を具体的にお願いします。",
            "cn": "請具體說明您的問題。",
            "default": "질문하신 내용에 대해 지식 기반하에 답변을 생성할 수 없습니다. 질문을 구체적으로 해주세요."
        }

    def validate(self, request: ChatRequest) -> Optional[ChatResponse]:
        """
        요청의 검색 문서 유효성을 검증합니다.

        Args:
            request (ChatRequest): 검증할 채팅 요청

        Returns:
            Optional[ChatResponse]: 유효성 검증 실패 시 응답, 성공 시 None
        """
        session_id = request.meta.session_id

        # 문서 존재 여부 확인
        if not request.chat.payload:
            logger.debug(f"[{session_id}] 요청에 검색 문서가 없습니다")

            # 시스템 메시지 결정
            system_msg = self._system_messages.get(request.chat.lang, self._system_messages["default"])

            # 오류 응답 생성
            return ChatResponse(
                result_cd=ErrorCd.get_code(ErrorCd.SUCCESS_NO_DATA),
                result_desc=ErrorCd.get_description(ErrorCd.SUCCESS_NO_DATA),
                meta=MetaRes(
                    company_id=request.meta.company_id,
                    dept_class=request.meta.dept_class,
                    session_id=session_id,
                    rag_sys_info=request.meta.rag_sys_info,
                ),
                chat=ChatRes(
                    user=request.chat.user,
                    system=system_msg,
                    category1=request.chat.category1,
                    category2=request.chat.category2,
                    category3=request.chat.category3,
                    info=[]
                )
            )

        # 문서가 있으면 유효성 검증 통과
        return None
