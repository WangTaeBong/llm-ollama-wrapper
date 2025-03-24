"""
제공자 팩토리 모듈

다양한 데이터 제공자 인스턴스를 생성합니다.
"""

import logging
from typing import Dict, Type, Any

from src.services.retrieval.base import DataProviderBase, CacheManagerBase
from src.services.retrieval.providers.api_provider import APIProvider
from src.services.retrieval.providers.web_provider import WebSearchProvider

# 로거 설정
logger = logging.getLogger(__name__)


class ProviderFactory:
    """
    데이터 제공자 팩토리 클래스

    필요한 데이터 제공자 인스턴스를 생성합니다.
    """

    def __init__(self):
        """제공자 팩토리 초기화"""
        from src.common.config_loader import ConfigLoader
        self.settings = ConfigLoader().get_settings()

    def create_provider(self, provider_type: str, request_data: Any,
                        cache_manager: CacheManagerBase) -> DataProviderBase:
        """
        데이터 제공자 인스턴스 생성

        Args:
            provider_type: 제공자 유형 ("api" 또는 "web")
            request_data: 요청 데이터
            cache_manager: 캐시 관리자

        Returns:
            DataProviderBase: 생성된 데이터 제공자 인스턴스

        Raises:
            ValueError: 지원되지 않는 제공자 유형
        """
        if provider_type == "api":
            return APIProvider(
                url=self.settings.api.retrival_api,
                headers={"content-type": "application/json;charset=utf-8"},
                cache_manager=cache_manager
            )
        elif provider_type == "web":
            return WebSearchProvider(
                settings=self.settings,
                cache_manager=cache_manager
            )
        else:
            raise ValueError(f"지원되지 않는 제공자 유형: {provider_type}")
