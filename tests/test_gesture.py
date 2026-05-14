from src.gesture import GestureAnalyzer


def test_speed_is_zero_for_first_point():
    analyzer = GestureAnalyzer(fast_speed_threshold=100.0)

    state = analyzer.update((0, 0), timestamp=1.0)

    assert state.speed == 0.0
    assert state.is_fast is False


def test_speed_and_fast_state_are_calculated():
    analyzer = GestureAnalyzer(fast_speed_threshold=100.0)

    analyzer.update((0, 0), timestamp=1.0)
    state = analyzer.update((30, 40), timestamp=1.5)

    assert state.speed == 100.0
    assert state.is_fast is True


def test_missing_point_resets_motion_state_without_crashing():
    analyzer = GestureAnalyzer(fast_speed_threshold=100.0)

    analyzer.update((0, 0), timestamp=1.0)
    state = analyzer.update(None, timestamp=1.5)

    assert state.speed == 0.0
    assert state.is_fast is False
