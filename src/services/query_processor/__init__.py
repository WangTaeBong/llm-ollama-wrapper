"""
쿼리 프로세서 패키지

사용자 쿼리의 전처리, 필터링, 패턴 감지 및 최적화 기능을 제공합니다.
"""

from src.services.query_processor.factory import ProcessorFactory
from src.services.query_processor.exceptions import QueryProcessorError
from src.services.query_processor.cache_manager import QueryCache


class QueryProcessor:
    """
    쿼리 프로세서 - 기존 API와의 호환성 제공

    기존 QueryProcessor 클래스의 API를 유지하면서 내부적으로
    새로운 아키텍처를 사용합니다.
    """

    def __init__(self, settings, query_check_json_dict):
        """
        QueryProcessor 초기화

        Args:
            settings: 설정 객체
            query_check_json_dict: 쿼리 패턴 사전
        """
        self.settings = settings
        self.query_check_json_dict = query_check_json_dict

        # 공유 캐시 인스턴스
        self.cache = QueryCache.get_instance()

        # 프로세서 팩토리를 통한 프로세서 인스턴스 생성
        self._factory = ProcessorFactory()

    @property
    def _standard_processor(self):
        """지연 초기화된 표준 프로세서"""
        return self._factory.create("standard", self.settings, self.query_check_json_dict)

    @property
    def _faq_processor(self):
        """지연 초기화된 FAQ 프로세서"""
        return self._factory.create("faq", self.settings, self.query_check_json_dict)

    @property
    def _pattern_processor(self):
        """지연 초기화된 패턴 프로세서"""
        return self._factory.create("pattern", self.settings, self.query_check_json_dict)

    def clean_query(self, query):
        """
        사용자 입력 쿼리에서 불필요한 특수 문자와 기호를 제거합니다.

        Args:
            query (str): 정제할 원본 쿼리

        Returns:
            str: 정제된 쿼리
        """
        return self._standard_processor.clean_query(query)

    def filter_query(self, query):
        """
        쿼리에서 불필요한 문자를 제거하기 위한 필터를 적용합니다.

        Args:
            query (str): 필터링할 원본 쿼리

        Returns:
            str: 필터링된 쿼리
        """
        return self._standard_processor.filter_query(query)

    def check_query_sentence(self, request):
        """
        사용자 쿼리를 미리 정의된 응답 패턴과 비교하여 적절한 응답을 생성합니다.

        Args:
            request: 처리할 채팅 요청 객체

        Returns:
            Optional[str]: 일치하는 패턴이 있으면 응답 문자열, 없으면 None
        """
        return self._pattern_processor.check_query_sentence(request)

    def construct_faq_query(self, request):
        """
        FAQ 카테고리 기반으로 최적화된 LLM 쿼리를 생성합니다.

        Args:
            request: 처리할 채팅 요청 객체

        Returns:
            str: 구성된 FAQ 쿼리 또는 원본 쿼리
        """
        return self._faq_processor.construct_faq_query(request)

    def reset_cache(self):
        """
        캐시를 초기화합니다.
        설정이 변경되었을 때 호출하세요.
        """
        self.cache.clear()
        ProcessorFactory.clear_instances()


__all__ = ['QueryProcessor', 'QueryProcessorError', 'ProcessorFactory', 'QueryCache']
