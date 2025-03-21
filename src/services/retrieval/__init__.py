"""
검색 시스템 패키지
===============

다양한 소스로부터 문서를 검색하고 처리하는 기능을 제공합니다.
"""

# 기본 인터페이스와 팩토리 임포트
from .base import RetrievalSourceBase
from .factory import RetrievalSourceFactory

# 구현체 임포트 - 팩토리에 자동 등록됨
from .sources.api_source import APIRetrievalSource
from .sources.web_source import WebRetrievalSource
from .sources.hybrid_source import HybridRetrievalSource

# 구현체 임포트 - 팩토리에 자동 등록됨

__all__ = [
    'RetrievalSourceBase',
    'RetrievalSourceFactory',
    'APIRetrievalSource',
    'WebRetrievalSource',
    'HybridRetrievalSource',
]
