"""
메시지 저장소 모듈

채팅 메시지 저장 및 검색 기능을 제공합니다.
"""

from typing import List, Dict, Any, Optional, Tuple


class RedisKeyGenerator:
    """
    Redis 키 생성 클래스

    Redis 저장소를 위한 키 생성 기능을 제공합니다.
    """

    @staticmethod
    def generate_key(*parts: str) -> str:
        """
        Redis 키를 생성합니다.

        여러 부분을 콜론(':')으로 연결합니다.

        Args:
            parts: 키 구성 요소

        Returns:
            str: 생성된 Redis 키
        """
        return ":".join(parts)

    @classmethod
    def chat_key(cls, system_info: str, session_id: str) -> str:
        """
        채팅 세션용 Redis 키를 생성합니다.

        Args:
            system_info: 시스템 정보
            session_id: 세션 ID

        Returns:
            str: 생성된 Redis 키
        """
        return cls.generate_key("chat", system_info, session_id)

    @classmethod
    def message_key(cls, system_info: str, session_id: str, message_id: str) -> str:
        """
        메시지용 Redis 키를 생성합니다.

        Args:
            system_info: 시스템 정보
            session_id: 세션 ID
            message_id: 메시지 ID

        Returns:
            str: 생성된 Redis 키
        """
        return cls.generate_key("message", system_info, session_id, message_id)
