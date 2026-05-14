import numpy as np

from src.trail import Trail


def test_trail_adds_and_expires_points():
    trail = Trail(duration_seconds=1.0)

    trail.add_point((10, 20), timestamp=0.0, speed=0.0)
    trail.add_point((20, 30), timestamp=0.5, speed=100.0)
    trail.prune(now=1.6)

    assert trail.points == [((20, 30), 0.5, 100.0)]


def test_trail_draw_changes_frame_pixels():
    trail = Trail(duration_seconds=1.0, color_bgr=(0, 255, 0))
    frame = np.zeros((80, 80, 3), dtype=np.uint8)

    trail.add_point((10, 10), timestamp=0.0, speed=0.0)
    trail.add_point((60, 60), timestamp=0.2, speed=50.0)
    output = trail.draw(frame.copy(), now=0.3)

    assert int(output.sum()) > 0
