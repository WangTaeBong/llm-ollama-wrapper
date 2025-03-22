"""
메시지 처리 핸들러 모듈

채팅 메시지의 처리 흐름을 관리합니다.
"""

import logging
from typing import List, Dict, Any, Optional

from src.services.messaging.models import ChatMessage, ChatSession
from src.services.messaging.formatters import MessageFormatter
from src.services.messaging.storage import RedisKeyGenerator
from src.utils.redis_utils import RedisUtils

# 로거 설정
logger = logging.getLogger(__name__)


class MessageHandler:
    """
    메시지 처리 핸들러 클래스

    채팅 메시지 처리 로직을 캡슐화합니다.
    """

    def __init__(self, system_info: str = None):
        """
        메시지 핸들러 초기화

        Args:
            system_info: 시스템 정보
        """
        self.system_info = system_info
        self.formatter = MessageFormatter()
        self.key_generator = RedisKeyGenerator()

    def create_message(self, role: str, content: str, timestamp: Optional[str] = None) -> Dict[str, str]:
        """
        새 메시지를 생성합니다.

        Args:
            role: 메시지 발신자 역할
            content: 메시지 내용
            timestamp: 타임스탬프 (기본값: 현재 시간)

        Returns:
            Dict[str, str]: 생성된 메시지
        """
        return self.formatter.create_message(role, content, timestamp)

    def create_chat_data(self, chat_id: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        채팅 데이터를 생성합니다.

        Args:
            chat_id: 채팅 ID
            messages: 메시지 목록

        Returns:
            Dict[str, Any]: 생성된 채팅 데이터
        """
        return self.formatter.create_chat_data(chat_id, messages)

    async def save_to_redis(self, system_info: str, session_id: str, chat_data: Dict[str, Any]) -> bool:
        """
        채팅 데이터를 Redis에 저장합니다.

        Args:
            system_info: 시스템 정보
            session_id: 세션 ID
            chat_data: 저장할 채팅 데이터

        Returns:
            bool: 성공 여부
        """
        try:
            # Redis 키 생성
            key = self.key_generator.chat_key(system_info or self.system_info, session_id)

            # Redis에 저장
            await RedisUtils.async_save_message_to_redis(
                system_info=system_info or self.system_info,
                session_id=session_id,
                message=chat_data
            )

            logger.debug(f"메시지가 성공적으로 Redis에 저장되었습니다. 키: {key}")
            return True

        except Exception as e:
            logger.error(f"Redis 저장 중 오류 발생: {e}")
            return False

    @staticmethod
    async def get_from_redis(system_info: str, session_id: str) -> List[Dict[str, Any]]:
        """
        Redis에서 채팅 데이터를 검색합니다.

        Args:
            system_info: 시스템 정보
            session_id: 세션 ID

        Returns:
            List[Dict[str, Any]]: 검색된 채팅 메시지
        """
        try:
            # Redis에서 메시지 검색
            messages = RedisUtils.get_messages_from_redis(
                system_info=system_info,
                session_id=session_id
            )

            return messages

        except Exception as e:
            logger.error(f"Redis 검색 중 오류 발생: {e}")
            return []
