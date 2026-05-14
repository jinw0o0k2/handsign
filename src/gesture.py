from __future__ import annotations

from dataclasses import dataclass
from math import hypot


@dataclass(frozen=True)
class GestureState:
    fingertip: tuple[int, int] | None
    speed: float
    is_fast: bool
    hand_open: bool = False
    fist: bool = False


class GestureAnalyzer:
    def __init__(self, fast_speed_threshold: float) -> None:
        self.fast_speed_threshold = fast_speed_threshold
        self._last_point: tuple[int, int] | None = None
        self._last_timestamp: float | None = None

    def update(self, point: tuple[int, int] | None, timestamp: float) -> GestureState:
        if point is None:
            self._last_point = None
            self._last_timestamp = None
            return GestureState(fingertip=None, speed=0.0, is_fast=False)

        speed = 0.0
        if self._last_point is not None and self._last_timestamp is not None:
            dt = timestamp - self._last_timestamp
            if dt > 0:
                dx = point[0] - self._last_point[0]
                dy = point[1] - self._last_point[1]
                speed = hypot(dx, dy) / dt

        self._last_point = point
        self._last_timestamp = timestamp
        return GestureState(
            fingertip=point,
            speed=speed,
            is_fast=speed >= self.fast_speed_threshold,
        )
