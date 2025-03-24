"""
메모리 기반 히스토리 저장소 모듈

메모리에 대화 히스토리를 저장하고 검색하는 기능을 제공합니다.
주로 테스트 용도 및 Redis가 사용 불가능한 환경을 위한 대체 구현입니다.
"""

import logging
import time
from typing import List, Dict, Any, DefaultDict, Optional
from collections import defaultdict

from src.services.history.base import HistoryStorageBase

# 로거 설정
logger = logging.getLogger(__name__)


class MemoryHistoryStorage(HistoryStorageBase):
    """
    메모리 기반 히스토리 저장소 클래스

    인메모리 저장소를 사용하여 대화 히스토리를 저장하고 검색합니다.
    주로 테스트 환경이나 임시 저장용으로 사용됩니다.
    """

    def __init__(self, ttl: int = 3600):
        """
        메모리 히스토리 저장소 초기화

        Args:
            ttl: 메시지 유효 시간(초), 기본값 1시간
        """
        self.ttl = ttl
        self._storage: DefaultDict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list))
        self._timestamps: DefaultDict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float))
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # 5분마다 자동 정리

    async def save_message(self, system_info: str, session_id: str, message: Dict[str, Any]) -> bool:
        """
        메시지를 메모리에 저장합니다.

        Args:
            system_info: 시스템 정보
            session_id: 세션 ID
            message: 저장할 메시지 데이터

        Returns:
            bool: 저장 성공 여부
        """
        try:
            # 정기 정리 확인
            self._check_cleanup()

            # 키 생성
            key = f"{system_info}:{session_id}"

            # 메시지 저장
            self._storage[system_info][session_id].append(message)
            self._timestamps[system_info][session_id] = time.time()

            return True
        except Exception as e:
            logger.error(f"[{session_id}] 메모리 저장소 저장 실패: {str(e)}")
            return False

    def get_messages(self, system_info: str, session_id: str) -> List[Dict[str, Any]]:
        """
        메모리에서 메시지를 가져옵니다.

        Args:
            system_info: 시스템 정보
            session_id: 세션 ID

        Returns:
            List[Dict[str, Any]]: 저장된 메시지 목록
        """
        try:
            # 정기 정리 확인
            self._check_cleanup()

            # 만료 확인
            timestamp = self._timestamps[system_info].get(session_id, 0)
            if time.time() - timestamp > self.ttl:
                logger.debug(f"[{session_id}] 메시지 만료로 빈 목록 반환")
                return []

            # 메시지 검색
            return self._storage[system_info].get(session_id, [])
        except Exception as e:
            logger.error(f"[{session_id}] 메모리 저장소 검색 실패: {str(e)}")
            return []

    async def get_messages_async(self, system_info: str, session_id: str) -> List[Dict[str, Any]]:
        """
        메모리에서 메시지를 비동기적으로 가져옵니다.
        메모리 저장소는 동기화 이슈가 없어 동기 메서드와 동일하게 처리합니다.

        Args:
            system_info: 시스템 정보
            session_id: 세션 ID

        Returns:
            List[Dict[str, Any]]: 저장된 메시지 목록
        """
        return self.get_messages(system_info, session_id)

    async def clear_messages(self, system_info: str, session_id: str) -> bool:
        """
        지정된 세션의 메시지를 모두 삭제합니다.

        Args:
            system_info: 시스템 정보
            session_id: 세션 ID

        Returns:
            bool: 삭제 성공 여부
        """
        try:
            if session_id in self._storage[system_info]:
                del self._storage[system_info][session_id]
            if session_id in self._timestamps[system_info]:
                del self._timestamps[system_info][session_id]
            return True
        except Exception as e:
            logger.error(f"[{session_id}] 메모리 저장소 삭제 실패: {str(e)}")
            return False

    def _check_cleanup(self) -> None:
        """
        오래된 항목을 정리합니다.
        주기적으로 자동 호출되어 메모리 효율성을 유지합니다.
        """
        current_time = time.time()

        # 정리 주기 확인
        if current_time - self._last_cleanup < self._cleanup_interval:
            return

        try:
            logger.debug("메모리 저장소 자동 정리 시작")
            cleaned_sessions = 0

            # 각 시스템의 만료된 세션 정리
            for system_info in list(self._timestamps.keys()):
                for session_id in list(self._timestamps[system_info].keys()):
                    timestamp = self._timestamps[system_info][session_id]
                    if current_time - timestamp > self.ttl:
                        # 만료된 세션 삭제
                        if session_id in self._storage[system_info]:
                            del self._storage[system_info][session_id]
                        del self._timestamps[system_info][session_id]
                        cleaned_sessions += 1

                # 비어있는 시스템 항목 정리
                if not self._timestamps[system_info]:
                    del self._timestamps[system_info]
                if not self._storage[system_info]:
                    del self._storage[system_info]

            self._last_cleanup = current_time
            if cleaned_sessions > 0:
                logger.debug(f"메모리 저장소 정리 완료: {cleaned_sessions}개 세션 제거됨")

        except Exception as e:
            logger.error(f"메모리 저장소 정리 중 오류: {str(e)}")
