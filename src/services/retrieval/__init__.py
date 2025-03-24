"""
검색 패키지

문서 검색과 관련된 클래스와 유틸리티를 제공합니다.
"""

from collections import defaultdict
import logging
import time
from typing import Any, List, Dict, Optional

from langchain_core.documents import Document

from src.schema.retriever_req import RetrieverRequest, RetrieverMeta, RetrieverQuery
from src.services.retrieval.custom_retriever import CustomRetriever
from src.services.retrieval.cache.cache_manager import CacheManager
from src.services.retrieval.document.document_store import DocumentStore

# 로거 설정
logger = logging.getLogger(__name__)


def create_retriever(request_data: Any) -> CustomRetriever:
    """
    커스텀 검색기를 생성하는 팩토리 함수

    기존 코드와의 호환성을 보장합니다.

    Args:
        request_data: 검색 요청 데이터

    Returns:
        CustomRetriever: 초기화된 검색기 인스턴스
    """
    if isinstance(request_data, RetrieverRequest):
        return CustomRetriever(request_data=request_data)
    else:
        # 요청 데이터가 RetrieverRequest 타입이 아닌 경우 변환
        try:
            if hasattr(request_data, 'meta') and hasattr(request_data, 'chat'):
                retriever_request = RetrieverRequest(
                    meta=RetrieverMeta(
                        company_id=request_data.meta.company_id,
                        dept_class=request_data.meta.dept_class,
                        rag_sys_info=request_data.meta.rag_sys_info,
                        session_id=request_data.meta.session_id,
                    ),
                    chat=RetrieverQuery(
                        user=request_data.chat.user,
                        category1=getattr(request_data.chat, 'category1', ''),
                        category2=getattr(request_data.chat, 'category2', ''),
                        category3=getattr(request_data.chat, 'category3', ''),
                    ),
                )
                return CustomRetriever(request_data=retriever_request)
            else:
                raise ValueError("요청 데이터 구조가 지원되지 않습니다")
        except Exception as e:
            logger.error(f"검색기 생성 중 오류: {e}")
            raise


__all__ = [
    'CustomRetriever',
    'create_retriever',
    'CacheManager',
    'DocumentStore'
]
