"""
메시지 포맷 모듈

채팅 메시지의 다양한 포맷 변환 기능을 제공합니다.
"""

from typing import Dict, List, Any, Optional
from src.services.messaging.models import ChatMessage, ChatSession
from src.services.messaging.utils import generate_timestamp


class MessageFormatter:
    """
    메시지 포맷 변환 클래스

    다양한 형식의 메시지 변환 기능을 제공합니다.
    """

    @staticmethod
    def create_message(role: str, content: str, timestamp: Optional[str] = None) -> Dict[str, str]:
        """
        메시지 딕셔너리를 생성합니다.

        Args:
            role: 메시지 발신자 역할 (예: "HumanMessage" 또는 "AIMessage")
            content: 메시지 내용
            timestamp: 메시지 타임스탬프 (기본값: 현재 시간)

        Returns:
            Dict[str, str]: 메시지 딕셔너리
        """
        return {
            "role": role,
            "content": content,
            "timestamp": timestamp or generate_timestamp()
        }

    @staticmethod
    def create_chat_data(chat_id: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        채팅 데이터 딕셔너리를 생성합니다.

        Args:
            chat_id: 채팅 세션 식별자
            messages: 메시지 딕셔너리 목록

        Returns:
            Dict[str, Any]: 채팅 데이터 딕셔너리
        """
        return {
            "chat_id": chat_id,
            "messages": [
                {
                    "role": message.get("role"),
                    "content": message.get("content"),
                    "timestamp": message.get("timestamp", generate_timestamp())
                }
                for message in messages
            ]
        }

    @classmethod
    def from_chat_session(cls, session: ChatSession) -> Dict[str, Any]:
        """
        ChatSession 객체를 딕셔너리로 변환합니다.

        Args:
            session: 변환할 ChatSession 객체

        Returns:
            Dict[str, Any]: 변환된 딕셔너리
        """
        return session.to_dict()
