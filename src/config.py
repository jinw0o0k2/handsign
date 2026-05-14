from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    camera_index: int = 0
    window_name: str = "Hand Trail Tracker"
    frame_width: int = 1280
    frame_height: int = 720
    trail_duration_seconds: float = 1.2
    trail_color_bgr: tuple[int, int, int] = (255, 120, 40)
    trail_core_thickness: int = 4
    trail_glow_thickness: int = 18
    fast_speed_threshold: float = 900.0
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.6
