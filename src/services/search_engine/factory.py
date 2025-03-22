"""
검색 엔진 팩토리 모듈

설정에 따라 적절한 검색 엔진 인스턴스를 생성하는 팩토리를 제공합니다.
"""

import logging
from typing import Dict, Any, Optional, Type

from src.services.search_engine.base import SearchEngineBase
from src.services.search_engine.duckduckgo_engine import DuckDuckGoEngine
from src.services.search_engine.exceptions import EngineNotFoundError

# 로거 설정
logger = logging.getLogger(__name__)


class SearchEngineFactory:
    """
    검색 엔진 팩토리 클래스

    설정에 따라 적절한 검색 엔진 인스턴스를 생성합니다.
    """

    _registry: Dict[str, Type[SearchEngineBase]] = {
        "duckduckgo": DuckDuckGoEngine,
    }

    @classmethod
    def register_engine(cls, name: str, engine_class: Type[SearchEngineBase]) -> None:
        """
        새로운 검색 엔진을 등록합니다.

        Args:
            name: 검색 엔진 식별자
            engine_class: 검색 엔진 클래스
        """
        cls._registry[name.lower()] = engine_class
        logger.info(f"검색 엔진 '{name}'을(를) 등록했습니다")

    @classmethod
    def create_engine(cls, engine_type: str, settings: Any) -> SearchEngineBase:
        """
        지정된 유형의 검색 엔진을 생성합니다.

        Args:
            engine_type: 검색 엔진 유형
            settings: 설정 객체

        Returns:
            SearchEngineBase: 생성된 검색 엔진 인스턴스

        Raises:
            EngineNotFoundError: 지정된 유형의 검색 엔진을 찾을 수 없는 경우
        """
        engine_type = engine_type.lower()

        if engine_type not in cls._registry:
            logger.error(f"알 수 없는 검색 엔진 유형: {engine_type}")
            raise EngineNotFoundError(f"알 수 없는 검색 엔진 유형: {engine_type}")

        engine_class = cls._registry[engine_type]
        logger.debug(f"'{engine_type}' 검색 엔진을 생성합니다")
        return engine_class(settings)

    @classmethod
    def create_default_engine(cls, settings: Any) -> SearchEngineBase:
        """
        기본 검색 엔진을 생성합니다.

        Args:
            settings: 설정 객체

        Returns:
            SearchEngineBase: 생성된 검색 엔진 인스턴스
        """
        # 설정에서 기본 엔진 유형 가져오기 또는 DuckDuckGo 사용
        default_engine = getattr(settings.web_search, 'engine', 'duckduckgo')
        return cls.create_engine(default_engine, settings)
    