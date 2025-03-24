"""
성능 추적 모듈

처리 단계별 성능 지표를 추적하고 관리하는 기능을 제공합니다.
"""

import time
from typing import Dict, Any, List, Optional


class PerformanceTracker:
    """
    성능 추적 클래스

    처리 단계별 소요 시간과 자원 사용량을 추적하여 성능 지표를 수집합니다.
    """

    def __init__(self, session_id: str = None):
        """
        성능 추적기 초기화

        Args:
            session_id: 추적 대상 세션 ID (선택)
        """
        self.session_id = session_id
        self.stages = {}
        self.start_time = None
        self.stage_start_time = None
        self.current_stage = None

    def start_tracking(self) -> None:
        """전체 추적 시작"""
        self.start_time = time.time()

    def start_stage(self, stage_name: str) -> None:
        """
        특정 단계 추적 시작

        Args:
            stage_name: 추적할 단계 이름
        """
        self.stage_start_time = time.time()
        self.current_stage = stage_name

    def end_stage(self, stage_name: Optional[str] = None) -> float:
        """
        단계 추적 종료 및 소요 시간 기록

        Args:
            stage_name: 종료할 단계 이름 (기본값: 현재 단계)

        Returns:
            float: 단계 소요 시간(초)
        """
        if self.stage_start_time is None:
            return 0

        stage = stage_name or self.current_stage
        if stage is None:
            return 0

        elapsed = time.time() - self.stage_start_time
        self.stages[stage] = elapsed
        self.current_stage = None
        self.stage_start_time = None

        return elapsed

    def record_stage(self, stage_name: str, duration: float) -> None:
        """
        단계 소요 시간을 직접 기록

        Args:
            stage_name: 단계 이름
            duration: 소요 시간(초)
        """
        self.stages[stage_name] = duration

    def get_stage_time(self, stage_name: str) -> float:
        """
        특정 단계의 소요 시간 조회

        Args:
            stage_name: 조회할 단계 이름

        Returns:
            float: 단계 소요 시간(초) 또는 0 (미측정 시)
        """
        return self.stages.get(stage_name, 0)

    def get_total_elapsed(self) -> float:
        """
        전체 소요 시간 계산

        Returns:
            float: 전체 소요 시간(초)
        """
        if self.start_time is None:
            return 0

        # 모든 단계가 명시적으로 측정된 경우 합산
        if self.stages:
            return sum(self.stages.values())

        # 단계 측정이 없으면 전체 시간 반환
        return time.time() - self.start_time

    def get_stages(self) -> Dict[str, float]:
        """
        모든 단계 소요 시간 조회

        Returns:
            Dict[str, float]: 단계별 소요 시간
        """
        return self.stages

    def get_metrics(self) -> Dict[str, Any]:
        """
        성능 지표 조회

        Returns:
            Dict[str, Any]: 성능 지표
        """
        total_time = self.get_total_elapsed()

        metrics = {
            "total_processing_time": total_time,
            "processing_stages": self.get_stages(),
            "avg_stage_time": total_time / len(self.stages) if self.stages else 0,
        }

        return metrics

    def reset(self) -> None:
        """모든 측정 데이터 초기화"""
        self.stages.clear()
        self.start_time = None
        self.stage_start_time = None
        self.current_stage = None
