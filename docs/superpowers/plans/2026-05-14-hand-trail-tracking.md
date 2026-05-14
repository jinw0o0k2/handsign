# Hand Trail Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python desktop app that tracks the index fingertip with MediaPipe and draws a glowing OpenCV trail that fades over time.

**Architecture:** The app is split into small modules: camera access, hand landmark tracking, trail state/rendering, gesture state, effects, and the main app loop. Pure logic modules are tested first; webcam and MediaPipe integration are wired after the boundaries are stable.

**Tech Stack:** Python 3.10+, OpenCV, MediaPipe, NumPy, pytest.

---

## File Structure

- Create: `requirements.txt`
- Create: `main.py`
- Create: `src/__init__.py`
- Create: `src/config.py`
- Create: `src/camera.py`
- Create: `src/hand_tracker.py`
- Create: `src/trail.py`
- Create: `src/gesture.py`
- Create: `src/effects.py`
- Create: `src/app.py`
- Create: `tests/test_trail.py`
- Create: `tests/test_gesture.py`
- Create: `tests/test_effects.py`

## Task 1: Project Scaffold

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `src/config.py`

- [ ] **Step 1: Add dependencies**

Create `requirements.txt`:

```text
opencv-python>=4.9.0
mediapipe>=0.10.14
numpy>=1.26.0
pytest>=8.0.0
```

- [ ] **Step 2: Create package marker**

Create `src/__init__.py`:

```python
"""Hand trail tracking package."""
```

- [ ] **Step 3: Add shared configuration**

Create `src/config.py`:

```python
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
```

- [ ] **Step 4: Run a syntax check**

Run: `python -m py_compile src/config.py`

Expected: exits with no output.

- [ ] **Step 5: Commit scaffold**

```bash
git add requirements.txt src/__init__.py src/config.py
git commit -m "feat: add project scaffold"
```

## Task 2: Trail Logic and Rendering

**Files:**
- Create: `tests/test_trail.py`
- Create: `src/trail.py`

- [ ] **Step 1: Write failing trail tests**

Create `tests/test_trail.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_trail.py -v`

Expected: FAIL with `ModuleNotFoundError` or missing `Trail`.

- [ ] **Step 3: Implement trail**

Create `src/trail.py`:

```python
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
        cutoff = now - self.duration_seconds
        self.points = [point for point in self.points if point[1] >= cutoff]

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
            core_color = tuple(min(255, int(channel * (0.6 + alpha * 0.4))) for channel in self.color_bgr)
            thickness = max(1, int(self.core_thickness * alpha))

            cv2.line(glow_layer, start_pos, end_pos, glow_color, self.glow_thickness, cv2.LINE_AA)
            cv2.line(core_layer, start_pos, end_pos, core_color, thickness, cv2.LINE_AA)

        frame = cv2.addWeighted(frame, 1.0, glow_layer, 0.35, 0)
        frame = cv2.addWeighted(frame, 1.0, core_layer, 0.9, 0)
        return frame
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_trail.py -v`

Expected: PASS.

- [ ] **Step 5: Commit trail**

```bash
git add tests/test_trail.py src/trail.py
git commit -m "feat: add fading trail"
```

## Task 3: Gesture State

**Files:**
- Create: `tests/test_gesture.py`
- Create: `src/gesture.py`

- [ ] **Step 1: Write failing gesture tests**

Create `tests/test_gesture.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gesture.py -v`

Expected: FAIL with `ModuleNotFoundError` or missing `GestureAnalyzer`.

- [ ] **Step 3: Implement gesture analyzer**

Create `src/gesture.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gesture.py -v`

Expected: PASS.

- [ ] **Step 5: Commit gesture analyzer**

```bash
git add tests/test_gesture.py src/gesture.py
git commit -m "feat: add gesture motion state"
```

## Task 4: Effect Boundary

**Files:**
- Create: `tests/test_effects.py`
- Create: `src/effects.py`

- [ ] **Step 1: Write failing effect tests**

Create `tests/test_effects.py`:

```python
import numpy as np

from src.effects import EffectEngine
from src.gesture import GestureState


def test_effect_engine_returns_frame_shape_unchanged():
    engine = EffectEngine()
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    state = GestureState(fingertip=(1, 2), speed=10.0, is_fast=False)

    output = engine.apply(frame.copy(), state)

    assert output.shape == frame.shape
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_effects.py -v`

Expected: FAIL with missing `EffectEngine`.

- [ ] **Step 3: Implement effect engine**

Create `src/effects.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_effects.py -v`

Expected: PASS.

- [ ] **Step 5: Commit effect boundary**

```bash
git add tests/test_effects.py src/effects.py
git commit -m "feat: add effect engine boundary"
```

## Task 5: Camera and Hand Tracker

**Files:**
- Create: `src/camera.py`
- Create: `src/hand_tracker.py`

- [ ] **Step 1: Implement camera wrapper**

Create `src/camera.py`:

```python
from __future__ import annotations

import cv2
import numpy as np


class Camera:
    def __init__(self, index: int, width: int, height: int) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.capture = cv2.VideoCapture(index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def is_opened(self) -> bool:
        return bool(self.capture.isOpened())

    def read(self) -> tuple[bool, np.ndarray | None]:
        return self.capture.read()

    def release(self) -> None:
        self.capture.release()
```

- [ ] **Step 2: Implement MediaPipe hand tracker**

Create `src/hand_tracker.py`:

```python
from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np


class HandTracker:
    def __init__(
        self,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ) -> None:
        self._hands_module = mp.solutions.hands
        self._hands = self._hands_module.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def index_fingertip(self, frame_bgr: np.ndarray) -> tuple[int, int] | None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(frame_rgb)
        if not results.multi_hand_landmarks:
            return None

        landmarks = results.multi_hand_landmarks[0].landmark
        tip = landmarks[self._hands_module.HandLandmark.INDEX_FINGER_TIP]
        height, width = frame_bgr.shape[:2]
        x = min(width - 1, max(0, int(tip.x * width)))
        y = min(height - 1, max(0, int(tip.y * height)))
        return (x, y)

    def close(self) -> None:
        self._hands.close()
```

- [ ] **Step 3: Run syntax checks**

Run: `python -m py_compile src/camera.py src/hand_tracker.py`

Expected: exits with no output.

- [ ] **Step 4: Commit camera and tracker**

```bash
git add src/camera.py src/hand_tracker.py
git commit -m "feat: add camera and hand tracker"
```

## Task 6: App Loop and Entrypoint

**Files:**
- Create: `src/app.py`
- Create: `main.py`

- [ ] **Step 1: Implement application loop**

Create `src/app.py`:

```python
from __future__ import annotations

import time

import cv2

from src.camera import Camera
from src.config import AppConfig
from src.effects import EffectEngine
from src.gesture import GestureAnalyzer
from src.hand_tracker import HandTracker
from src.trail import Trail


class HandTrailApp:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.camera = Camera(config.camera_index, config.frame_width, config.frame_height)
        self.tracker = HandTracker(
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        self.trail = Trail(
            duration_seconds=config.trail_duration_seconds,
            color_bgr=config.trail_color_bgr,
            core_thickness=config.trail_core_thickness,
            glow_thickness=config.trail_glow_thickness,
        )
        self.gestures = GestureAnalyzer(config.fast_speed_threshold)
        self.effects = EffectEngine()

    def run(self) -> int:
        if not self.camera.is_opened():
            print(f"Could not open webcam at index {self.config.camera_index}.")
            self.close()
            return 1

        try:
            while True:
                ok, frame = self.camera.read()
                if not ok or frame is None:
                    print("Could not read frame from webcam.")
                    return 1

                frame = cv2.flip(frame, 1)
                now = time.monotonic()
                fingertip = self.tracker.index_fingertip(frame)
                state = self.gestures.update(fingertip, now)

                if fingertip is not None:
                    self.trail.add_point(fingertip, now, state.speed)

                frame = self.trail.draw(frame, now)
                frame = self.effects.apply(frame, state)

                cv2.imshow(self.config.window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return 0
        finally:
            self.close()

    def close(self) -> None:
        self.tracker.close()
        self.camera.release()
        cv2.destroyAllWindows()
```

- [ ] **Step 2: Implement entrypoint**

Create `main.py`:

```python
from src.app import HandTrailApp
from src.config import AppConfig


def main() -> int:
    app = HandTrailApp(AppConfig())
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run unit tests and syntax checks**

Run: `pytest -v`

Expected: all tests pass.

Run: `python -m py_compile main.py src/app.py`

Expected: exits with no output.

- [ ] **Step 4: Commit app loop**

```bash
git add main.py src/app.py
git commit -m "feat: wire hand trail app"
```

## Task 7: Manual Verification and README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Add beginner-friendly README**

Create `README.md`:

```markdown
# HandSign Hand Trail Tracker

This project uses Python, OpenCV, and MediaPipe to track the index fingertip from a webcam and draw a glowing trail that fades over time.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Press `q` in the camera window to quit.

## First Version

- Tracks the index fingertip.
- Draws a glowing trail.
- Fades old trail points.
- Calculates fingertip speed internally.
- Keeps gesture and effect modules ready for future expansion.

## Future Ideas

- Change the background when fingertip speed crosses a threshold.
- Detect open hand or fist gestures.
- Recognize drawn words and trigger visual effects.
```

- [ ] **Step 2: Run automated verification**

Run: `pytest -v`

Expected: all tests pass.

- [ ] **Step 3: Run manual app verification**

Run: `python main.py`

Expected:

- A camera window opens.
- Moving the index fingertip leaves a glowing trail.
- Removing the hand lets the trail fade away.
- Pressing `q` exits cleanly.

- [ ] **Step 4: Commit README**

```bash
git add README.md
git commit -m "docs: add setup and run instructions"
```

## Final Verification

- [ ] Run: `git status --short`

Expected: no uncommitted implementation files unless manual verification notes were intentionally left uncommitted.

- [ ] Run: `pytest -v`

Expected: all tests pass.

- [ ] Run: `python main.py`

Expected: manual webcam behavior matches the design spec.
