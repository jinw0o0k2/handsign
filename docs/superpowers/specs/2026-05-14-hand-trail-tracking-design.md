# Hand Trail Tracking Design

Date: 2026-05-14

## Goal

Build a beginner-friendly desktop computer vision project that uses a webcam, MediaPipe, and OpenCV to track the user's index fingertip and draw a glowing trail that fades over time.

The first version should stay simple on screen, but the code should be structured so gesture triggers and visual effects can be added later without rewriting the tracking logic.

## First Version Scope

The first version includes:

- A Python desktop app launched with `python main.py`.
- Webcam input through OpenCV.
- Hand landmark detection through MediaPipe.
- Index fingertip tracking using the MediaPipe hand landmark for the index finger tip.
- A glowing trail that follows the fingertip and fades naturally.
- Basic fingertip speed calculation for future trigger logic.
- Gesture and effect modules with minimal working interfaces for future expansion.
- Graceful behavior when no hand is visible.

The first version does not include:

- Drawing letters or recognizing handwritten words.
- Triggering effects from specific Korean words such as "사랑".
- Full fist/open-hand classification.
- Advanced particle effects.
- A GUI beyond the OpenCV camera window.

## Project Structure

```text
image_processing/
  main.py
  requirements.txt
  src/
    app.py
    camera.py
    hand_tracker.py
    trail.py
    gesture.py
    effects.py
    config.py
```

## Module Responsibilities

`main.py` starts the application.

`src/app.py` owns the main runtime loop. It connects camera input, hand tracking, trail drawing, gesture state, effect handling, and window display.

`src/camera.py` opens the webcam, reads frames, and releases camera resources.

`src/hand_tracker.py` wraps MediaPipe Hands. It detects hand landmarks and returns the index fingertip position in pixel coordinates when a hand is visible.

`src/trail.py` stores trail points with timestamps, removes expired points, and draws the fading glowing trail onto a frame.

`src/gesture.py` calculates motion state such as fingertip speed. It also provides a stable place to add open-hand, fist, or custom gesture detection later.

`src/effects.py` defines the effect system boundary. The initial implementation returns the frame unchanged through a clear `apply` interface, and future code can add speed-triggered background changes, gesture effects, or word-based effects here.

`src/config.py` contains tunable settings such as camera index, trail duration, trail color, line thickness, speed thresholds, and MediaPipe detection confidence.

## Frame Data Flow

Each camera frame follows this sequence:

```text
Read camera frame
-> Mirror frame horizontally
-> Detect hand landmarks with MediaPipe
-> Extract index fingertip position
-> Calculate speed from previous fingertip position
-> Add fingertip point to the trail
-> Remove expired trail points
-> Draw glowing fading trail
-> Pass frame and motion state through the effect system
-> Display final frame
```

If no hand is visible, the app does not add a new trail point. Existing trail points continue fading until they expire.

## Trail Behavior

The trail is represented as timestamped points. Each point has:

- Pixel position.
- Creation time.
- Optional speed value for future visual variation.

During rendering, older points become more transparent and thinner. The trail should feel like a soft glowing afterimage, not a hard technical line.

The implementation should keep this simple by drawing multiple OpenCV overlay passes with alpha blending. This avoids bringing in a heavier graphics framework for the first version.

## Extension Points

The design keeps three future expansion points clear:

- Speed triggers: `gesture.py` can mark motion as fast when fingertip speed crosses a threshold. `effects.py` can react by changing the background or trail style.
- Hand pose triggers: `gesture.py` can later classify open hand, fist, or other hand poses from landmarks.
- Word/effect triggers: a future module can collect drawn trail strokes, recognize a word or command, and ask `effects.py` to apply a matching visual effect.

These future features should consume tracker and trail state through clean interfaces rather than modifying MediaPipe or drawing code directly.

## Error Handling

The app should handle common beginner-project failures clearly:

- If the webcam cannot open, print a helpful message and exit.
- If a frame cannot be read, stop the loop safely.
- If MediaPipe detects no hand, continue running and fade the existing trail.
- If landmarks are missing or malformed, skip that frame's tracking update.
- On exit, release the camera and destroy OpenCV windows.

## User Controls

The initial app uses one keyboard control:

- `q`: quit the app.

Additional controls can be added later, such as clearing the trail, changing colors, or toggling effects.

## Verification

Manual verification for the first version:

- Install dependencies with `pip install -r requirements.txt`.
- Run the app with `python main.py`.
- Confirm the webcam window opens.
- Move the index fingertip in front of the camera and confirm a glowing trail follows it.
- Move the hand out of frame and confirm the trail fades away without errors.
- Press `q` and confirm the app exits cleanly.

Basic automated tests can be added later for pure logic, especially trail expiry and speed calculation. Webcam and MediaPipe behavior will mostly be verified manually in the first version.
