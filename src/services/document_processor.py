import logging
from typing import List, Optional

from langchain_core.documents import Document

from src.common.error_cd import ErrorCd
from src.schema.chat_req import ChatRequest, PayloadReq
from src.schema.chat_res import ChatResponse, MetaRes, ChatRes

# Configure logger with name of current module
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    A class responsible for document processing operations.

    This class provides functionality for converting, validating, and processing
    documents, facilitating the transformation between chat request objects and
    document objects.
    """

    def __init__(self, settings):
        """
        Initialize the DocumentProcessor class.

        Args:
            settings: An object containing configuration information
        """
        self.settings = settings

    @classmethod
    def convert_payload_to_document(cls, request: ChatRequest) -> List[Document]:
        """
        Extract valid documents from a ChatRequest and convert them to a list of Document objects.

        This method processes the payload from a chat request and transforms each valid
        document entry into a Document object with appropriate metadata.

        Args:
            request (ChatRequest): The chat request object to extract documents from

        Returns:
            List[Document]: A list of converted Document objects
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
            logger.warning(f"[{request.meta.session_id}] Payload to document conversion failed: {e}")
            return []

    @classmethod
    def convert_document_to_payload(cls, documents: List[Document]) -> List[PayloadReq]:
        """
        Convert a list of Document objects to a list of PayloadReq objects.

        This method transforms Document objects back into the payload format
        required for chat requests.

        Args:
            documents (List[Document]): The list of Document objects to convert

        Returns:
            List[PayloadReq]: A list of converted PayloadReq objects
        """
        if not documents:
            return []

        try:
            return [
                PayloadReq(
                    doc_name=doc.metadata.get("source", ""),  # Use correct key "source" instead of "doc_name"
                    doc_page=doc.metadata.get("doc_page", ""),
                    content=doc.page_content
                )
                for doc in documents if doc
            ]
        except Exception as e:
            # We don't have access to session_id here as it's not in the Document objects
            # Consider passing session_id as a parameter if needed for consistent logging
            logger.warning(f"Document to payload conversion failed: {e}")
            return []

    @classmethod
    def validate_retrieval_documents(cls, request: ChatRequest) -> Optional[ChatResponse]:
        """
        Validate the existence of retrieval documents in a chat request.

        This method checks if the request contains any documents and generates
        an appropriate response if no documents are found.

        Args:
            request (ChatRequest): The chat request object to validate

        Returns:
            Optional[ChatResponse]: An error response if no documents exist, None otherwise
        """
        # Predefined multilingual system messages
        system_messages = {
            "en": "We're unable to generate an answer based on our knowledge base for your question. "
                  "Please try to be more specific with your question.",
            "jp": "質問された内容について、ナレッジベースで回答を生成することができません。 質問を具体的にお願いします。",
            "cn": "請具體說明您的問題。",
            "default": "질문하신 내용에 대해 지식 기반하에 답변을 생성할 수 없습니다. 질문을 구체적으로 해주세요."
        }

        # Check if documents exist (removed unnecessary try-except)
        if not request.chat.payload:
            system_msg = system_messages.get(request.chat.lang, system_messages["default"])
            logger.debug(f"[{request.meta.session_id}] No retrieval documents found for the request")

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
