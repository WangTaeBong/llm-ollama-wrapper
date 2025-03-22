"""
쿼리 프로세서 팩토리 모듈

설정에 따라 적절한 쿼리 프로세서 인스턴스를 생성하는 팩토리를 제공합니다.
"""

import logging
from typing import Dict, Type, Any

from src.services.query_processor.base import QueryProcessorBase
from src.services.query_processor.processors.standard_processor import StandardQueryProcessor
from src.services.query_processor.processors.faq_processor import FAQQueryProcessor
from src.services.query_processor.processors.pattern_processor import PatternQueryProcessor
from src.services.query_processor.exceptions import ProcessorNotFoundError

# 로거 설정
logger = logging.getLogger(__name__)


class QueryProcessorFactory:
    """
    쿼리 프로세서 팩토리 클래스

    설정에 따라 적절한 쿼리 프로세서 인스턴스를 생성합니다.
    """

    _registry: Dict[str, Type[QueryProcessorBase]] = {
        "standard": StandardQueryProcessor,
        "faq": FAQQueryProcessor,
        "pattern": PatternQueryProcessor,
    }

    @classmethod
    def register_processor(cls, name: str, processor_class: Type[QueryProcessorBase]) -> None:
        """
        새로운 쿼리 프로세서를 등록합니다.

        Args:
            name: 프로세서 식별자
            processor_class: 프로세서 클래스
        """
        cls._registry[name.lower()] = processor_class
        logger.info(f"쿼리 프로세서 '{name}'을(를) 등록했습니다")

    def create_processor(self, processor_type: str, settings: Any,
                         query_check_json_dict=None) -> QueryProcessorBase:
        """
        지정된 유형의 쿼리 프로세서를 생성합니다.

        Args:
            processor_type: 프로세서 유형
            settings: 설정 객체
            query_check_json_dict: 쿼리 패턴 사전 (선택 사항)

        Returns:
            QueryProcessorBase: 생성된 쿼리 프로세서 인스턴스

        Raises:
            ProcessorNotFoundError: 지정된 유형의 프로세서를 찾을 수 없는 경우
        """
        processor_type = processor_type.lower()

        if processor_type not in self._registry:
            logger.error(f"알 수 없는 쿼리 프로세서 유형: {processor_type}")
            raise ProcessorNotFoundError(f"알 수 없는 쿼리 프로세서 유형: {processor_type}")

        processor_class = self._registry[processor_type]
        logger.debug(f"'{processor_type}' 쿼리 프로세서를 생성합니다")
        return processor_class(settings, query_check_json_dict)

    def create_default_processor(self, settings: Any,
                                 query_check_json_dict=None) -> QueryProcessorBase:
        """
        기본 쿼리 프로세서를 생성합니다.

        Args:
            settings: 설정 객체
            query_check_json_dict: 쿼리 패턴 사전 (선택 사항)

        Returns:
            QueryProcessorBase: 생성된 쿼리 프로세서 인스턴스
        """
        return self.create_processor("standard", settings, query_check_json_dict)
