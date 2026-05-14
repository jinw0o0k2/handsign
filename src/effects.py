from __future__ import annotations

import cv2
import numpy as np

from src.gesture import GestureState


class EffectEngine:
    def apply(self, frame: np.ndarray, state: GestureState) -> np.ndarray:
        if state.is_fast:
            tint = np.full_like(frame, (20, 10, 0))
            return cv2.addWeighted(frame, 1.0, tint, 0.25, 0)
        return frame
