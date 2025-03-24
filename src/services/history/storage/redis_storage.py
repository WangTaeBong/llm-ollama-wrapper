"""
Redis 기반 히스토리 저장소 모듈

Redis를 사용하여 대화 히스토리를 저장하고 검색하는 기능을 제공합니다.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional

from src.services.history.base import HistoryStorageBase
from src.utils.redis_utils import RedisUtils

# 로거 설정
logger = logging.getLogger(__name__)


class RedisHistoryStorage(HistoryStorageBase):
    """
    Redis 기반 히스토리 저장소 클래스

    Redis를 사용하여 대화 히스토리를 저장하고 검색합니다.
    비동기 작업 지원 및 성능 최적화 기능을 제공합니다.
    """

    def __init__(self, ttl: int = 86400):
        """
        Redis 히스토리 저장소 초기화

        Args:
            ttl: 레코드 유지 시간(초), 기본값 24시간
        """
        self.ttl = ttl
        self._retry_attempts = 3
        self._retry_delay = 0.5

    async def save_message(self, system_info: str, session_id: str, message: Dict[str, Any]) -> bool:
        """
        메시지를 Redis에 저장합니다.

        개선된 오류 처리와 재시도 메커니즘을 포함합니다.

        Args:
            system_info: 시스템 정보 (RAG 시스템 정보)
            session_id: 세션 ID
            message: 저장할 메시지 데이터

        Returns:
            bool: 저장 성공 여부
        """
        retry_count = 0

        while retry_count < self._retry_attempts:
            try:
                logger.debug(f"[{session_id}] Redis에 메시지 저장 시도 (시도 {retry_count + 1})")

                # Redis에 저장
                await RedisUtils.async_save_message_to_redis(
                    system_info=system_info,
                    session_id=session_id,
                    message=message
                )

                logger.debug(f"[{session_id}] 메시지가 성공적으로 Redis에 저장됨")
                return True

            except Exception as e:
                retry_count += 1
                error_level = "warning" if retry_count < self._retry_attempts else "error"
                logger.log(
                    logging.getLevelName(error_level.upper()),
                    f"[{session_id}] Redis 저장 실패 (시도 {retry_count}): {str(e)}"
                )

                # 마지막 시도가 아니면 잠시 대기 후 재시도
                if retry_count < self._retry_attempts:
                    await asyncio.sleep(self._retry_delay * retry_count)  # 점진적 지연

        return False

    def get_messages(self, system_info: str, session_id: str) -> List[Dict[str, Any]]:
        """
        Redis에서 메시지를 가져옵니다.

        Args:
            system_info: 시스템 정보 (RAG 시스템 정보)
            session_id: 세션 ID

        Returns:
            List[Dict[str, Any]]: 저장된 메시지 목록
        """
        try:
            # Redis에서 메시지 검색
            messages = RedisUtils.get_messages_from_redis(
                system_info=system_info,
                session_id=session_id
            )

            return messages or []

        except Exception as e:
            logger.error(f"[{session_id}] Redis 검색 중 오류: {str(e)}")
            return []

    @classmethod
    async def get_messages_async(cls, system_info: str, session_id: str) -> List[Dict[str, Any]]:
        """
        Redis에서 메시지를 비동기적으로 가져옵니다.

        Args:
            system_info: 시스템 정보 (RAG 시스템 정보)
            session_id: 세션 ID

        Returns:
            List[Dict[str, Any]]: 저장된 메시지 목록
        """
        try:
            # 현재 RedisUtils에 비동기 조회 메서드가 없으므로 별도 스레드에서 실행
            messages = await asyncio.to_thread(
                RedisUtils.get_messages_from_redis,
                system_info=system_info,
                session_id=session_id
            )

            return messages or []

        except Exception as e:
            logger.error(f"[{session_id}] 비동기 Redis 검색 중 오류: {str(e)}")
            return []

    async def clear_messages(self, system_info: str, session_id: str) -> bool:
        """
        지정된 세션의 메시지를 모두 삭제합니다.

        Args:
            system_info: 시스템 정보 (RAG 시스템 정보)
            session_id: 세션 ID

        Returns:
            bool: 삭제 성공 여부
        """
        try:
            # RedisUtils에 clear 메서드가 없으므로 별도 로직 구현 필요
            # Redis 키 이름 규칙에 따라 키 생성 후 삭제
            # 참고: Redis 키 패턴은 RedisUtils 구현에 맞춰 조정 필요
            key = f"chat:{system_info}:{session_id}"
            success = await asyncio.to_thread(
                RedisUtils.redis_client.delete,
                key
            )

            return success > 0

        except Exception as e:
            logger.error(f"[{session_id}] Redis 레코드 삭제 중 오류: {str(e)}")
            return False
