"""
비동기 로깅 유틸리티 모듈

비동기 환경에서 효율적인 로깅을 지원하는 기능을 제공합니다.
"""

import asyncio
import logging
import sys
import time
from typing import Dict, Any, Optional, Set, List, Tuple

# 기본 로거 설정
logger = logging.getLogger(__name__)


class AsyncLogger:
    """
    비동기 로깅 시스템

    비동기 환경에서 로깅을 큐에 넣어 처리함으로써,
    메인 처리 흐름의 성능에 영향을 최소화합니다.
    """

    # 싱글톤 인스턴스
    _instance = None

    # 로깅 레벨 매핑
    _level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }

    @classmethod
    def get_instance(cls):
        """
        싱글톤 인스턴스 반환

        Returns:
            AsyncLogger: 로거 인스턴스
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """로거 초기화"""
        # 싱글톤 패턴 검사
        if AsyncLogger._instance is not None:
            raise RuntimeError("AsyncLogger는 싱글톤 클래스입니다. get_instance()를 사용하세요.")

        # 로그 큐 초기화
        self._queue = asyncio.Queue()
        self._task = None
        self._initialized = False

        # 로그 처리 지표
        self._stats = {
            "processed_logs": 0,
            "error_count": 0,
            "start_time": time.time()
        }

        # 처리된 세션 ID 추적
        self._processed_sessions: Set[str] = set()

    async def start(self) -> None:
        """로깅 작업 시작"""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._process_logs())
            self._initialized = True
            logger.debug("비동기 로깅 시스템 시작됨")

    async def log(self, level: str, message: str, **kwargs) -> None:
        """
        로그 항목 큐에 추가

        Args:
            level: 로그 레벨
            message: 로그 메시지
            **kwargs: 추가 로그 컨텍스트
        """
        # 로깅 시스템 초기화 확인 및 필요시 시작
        if not self._initialized:
            await self.start()

        # 로그 큐에 항목 추가
        await self._queue.put((level, message, kwargs))

        # 세션 ID가 있는 경우 추적
        session_id = kwargs.get('session_id')
        if session_id:
            self._processed_sessions.add(session_id)

    async def _process_logs(self) -> None:
        """로그 처리 작업 루프"""
        while True:
            try:
                # 큐에서 로그 항목 가져오기
                level, message, kwargs = await self._queue.get()

                # 로그 레벨 매핑
                log_level = self._level_map.get(level, logging.INFO)

                # 추가 컨텍스트에서 표준 매개변수 추출
                exc_info = kwargs.pop('exc_info', None)
                extra = kwargs

                # 세션 ID가 있으면 로거 이름에 추가
                session_id = kwargs.get('session_id')
                log_name = f"chat.{session_id}" if session_id else "chat"
                session_logger = logging.getLogger(log_name)

                # 로그 출력
                session_logger.log(log_level, message, exc_info=exc_info, extra=extra)

                # 작업 완료 표시
                self._queue.task_done()
                self._stats["processed_logs"] += 1

            except asyncio.CancelledError:
                # 작업 취소 처리
                logger.warning("로그 처리 작업이 취소되었습니다")
                break
            except Exception as e:
                # 로깅 자체에서 오류 발생 시
                self._stats["error_count"] += 1
                # 기본 로거로 출력
                logger.error(f"로그 처리 중 오류: {str(e)}", exc_info=True)
                # 오류가 있어도 계속 작업
                await asyncio.sleep(0.1)  # 짧은 대기 후 재시도

    async def flush(self) -> None:
        """모든 대기 중인 로그를 처리합니다."""
        if self._queue:
            await self._queue.join()

    async def close(self) -> None:
        """
        로깅 시스템을 안전하게 종료합니다.
        모든 로그를 처리한 후 태스크를 취소합니다.
        """
        if not self._task:
            return

        # 대기 중인 로그 모두 처리
        await self.flush()

        # 작업 취소
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass

        self._initialized = False
        self._task = None
        logger.debug("비동기 로깅 시스템 종료됨")

    def get_stats(self) -> Dict[str, Any]:
        """
        로깅 시스템 통계 조회

        Returns:
            Dict[str, Any]: 로깅 통계
        """
        current_time = time.time()
        elapsed = current_time - self._stats["start_time"]

        return {
            "processed_logs": self._stats["processed_logs"],
            "error_count": self._stats["error_count"],
            "uptime": elapsed,
            "logs_per_second": self._stats["processed_logs"] / elapsed if elapsed > 0 else 0,
            "queue_size": self._queue.qsize() if self._queue else 0,
            "sessions_count": len(self._processed_sessions)
        }

    def cleanup_session_tracking(self) -> None:
        """세션 추적 데이터를 정리합니다."""
        self._processed_sessions.clear()
