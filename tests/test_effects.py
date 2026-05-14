import numpy as np

from src.effects import EffectEngine
from src.gesture import GestureState


def test_effect_engine_returns_frame_shape_unchanged():
    engine = EffectEngine()
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    state = GestureState(fingertip=(1, 2), speed=10.0, is_fast=False)

    output = engine.apply(frame.copy(), state)

    assert output.shape == frame.shape
