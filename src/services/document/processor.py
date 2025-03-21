"""
문서 처리 모듈
===========

문서 변환 및 처리 기능을 제공합니다.

기능:
- 페이로드와 문서 간 변환
- 문서 검증 및 필터링
- 메타데이터 처리
"""

import logging
from typing import List, Optional

from langchain_core.documents import Document

from src.common.error_cd import ErrorCd
from src.schema.chat_req import ChatRequest, PayloadReq
from src.schema.chat_res import ChatResponse, MetaRes, ChatRes
from src.services.document.base import DocumentProcessorBase
from src.services.document.factory import DocumentProcessorFactory

# 로거 설정
logger = logging.getLogger(__name__)


class StandardDocumentProcessor(DocumentProcessorBase):
    """
    표준 문서 처리기 구현

    페이로드-문서 변환 및 문서 검증을 담당합니다.
    """

    def __init__(self, settings):
        """
        표준 문서 처리기 초기화

        Args:
            settings: 설정 객체
        """
        super().__init__(settings)
        self.settings = settings

    @classmethod
    async def initialize(cls) -> bool:
        """
        문서 처리기 초기화

        Returns:
            bool: 초기화 성공 여부
        """
        # 특별한 초기화 작업 없음
        return True

    @classmethod
    async def convert_payload_to_document(cls, request: ChatRequest) -> List[Document]:
        """
        페이로드를 문서로 변환

        Args:
            request: 채팅 요청 객체

        Returns:
            List[Document]: 변환된 문서 리스트
        """
        if not request.chat.payload:
            return []

        try:
            return [
                Document(
                    page_content=doc.content,
                    metadata={
                        "source": doc.doc_name,
                        "doc_page": doc.doc_page
                    }
                )
                for doc in request.chat.payload if doc and doc.content
            ]
        except Exception as e:
            logger.warning(f"[{request.meta.session_id}] 페이로드를 문서로 변환 중 오류: {e}")
            return []

    @classmethod
    async def convert_document_to_payload(cls, documents: List[Document]) -> List[PayloadReq]:
        """
        문서를 페이로드로 변환

        Args:
            documents: 문서 리스트

        Returns:
            List[PayloadReq]: 변환된 페이로드 리스트
        """
        if not documents:
            return []

        try:
            return [
                PayloadReq(
                    doc_name=doc.metadata.get("source", ""),
                    doc_page=doc.metadata.get("doc_page", ""),
                    content=doc.page_content
                )
                for doc in documents if doc
            ]
        except Exception as e:
            logger.warning(f"문서를 페이로드로 변환 중 오류: {e}")
            return []

    @classmethod
    async def validate_retrieval_documents(cls, request: ChatRequest) -> Optional[ChatResponse]:
        """
        검색 문서 유효성 검증

        Args:
            request: 채팅 요청 객체

        Returns:
            Optional[ChatResponse]: 오류 응답 또는 None
        """
        # 다국어 시스템 메시지
        system_messages = {
            "en": "We're unable to generate an answer based on our knowledge base for your question. "
                  "Please try to be more specific with your question.",
            "jp": "質問された内容について、ナレッジベースで回答を生成することができません。 質問を具体的にお願いします。",
            "cn": "請具體說明您的問題。",
            "default": "질문하신 내용에 대해 지식 기반하에 답변을 생성할 수 없습니다. 질문을 구체적으로 해주세요."
        }

        # 문서 존재 여부 확인
        if not request.chat.payload:
            system_msg = system_messages.get(request.chat.lang, system_messages["default"])
            logger.debug(f"[{request.meta.session_id}] 검색 문서가 없습니다")

            return ChatResponse(
                result_cd=ErrorCd.get_code(ErrorCd.SUCCESS_NO_DATA),
                result_desc=ErrorCd.get_description(ErrorCd.SUCCESS_NO_DATA),
                meta=MetaRes(
                    company_id=request.meta.company_id,
                    dept_class=request.meta.dept_class,
                    session_id=request.meta.session_id,
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

        return None


# 클래스 등록
DocumentProcessorFactory.register_processor("standard", StandardDocumentProcessor)
