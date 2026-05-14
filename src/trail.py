from __future__ import annotations

import cv2
import numpy as np


TrailPoint = tuple[tuple[int, int], float, float]


class Trail:
    def __init__(
        self,
        duration_seconds: float,
        color_bgr: tuple[int, int, int] = (255, 120, 40),
        core_thickness: int = 4,
        glow_thickness: int = 18,
    ) -> None:
        self.duration_seconds = duration_seconds
        self.color_bgr = color_bgr
        self.core_thickness = core_thickness
        self.glow_thickness = glow_thickness
        self.points: list[TrailPoint] = []

    def add_point(self, point: tuple[int, int], timestamp: float, speed: float) -> None:
        self.points.append((point, timestamp, speed))

    def prune(self, now: float) -> None:
        if not self.points:
            return

        cutoff = now - self.duration_seconds
        latest_point = self.points[-1]
        self.points = [point for point in self.points[:-1] if point[1] >= cutoff]
        self.points.append(latest_point)

    def draw(self, frame: np.ndarray, now: float) -> np.ndarray:
        self.prune(now)
        if len(self.points) < 2:
            return frame

        glow_layer = np.zeros_like(frame)
        core_layer = np.zeros_like(frame)

        for start, end in zip(self.points, self.points[1:]):
            start_pos, start_time, _ = start
            end_pos, end_time, _ = end
            age = max(0.0, now - end_time)
            alpha = max(0.0, 1.0 - age / self.duration_seconds)
            if alpha <= 0.0:
                continue

            glow_color = tuple(int(channel * alpha) for channel in self.color_bgr)
            core_color = tuple(
                min(255, int(channel * (0.6 + alpha * 0.4)))
                for channel in self.color_bgr
            )
            thickness = max(1, int(self.core_thickness * alpha))

            cv2.line(
                glow_layer,
                start_pos,
                end_pos,
                glow_color,
                self.glow_thickness,
                cv2.LINE_AA,
            )
            cv2.line(core_layer, start_pos, end_pos, core_color, thickness, cv2.LINE_AA)

        frame = cv2.addWeighted(frame, 1.0, glow_layer, 0.35, 0)
        frame = cv2.addWeighted(frame, 1.0, core_layer, 0.9, 0)
        return frame
