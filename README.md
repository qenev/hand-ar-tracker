# Hand AR Tracker

Hand AR Tracker is a real-time hand and finger tracking augmented reality system built with Python. It uses a standard webcam feed processed through Google MediaPipe and OpenCV to detect, track, and visualize hand landmarks in real time. The system renders a green skeletal mesh over detected hands, marks each of the 21 keypoints with red dots, and supports gesture recognition for five distinct hand gestures. It is designed to run on both CPU and GPU configurations and is fully configurable through a YAML configuration file.

---

## Features

- Dual hand tracking with simultaneous detection and rendering of up to two hands
- 21 keypoints per hand following the MediaPipe hand landmark model
- Green skeletal mesh overlay connecting keypoints along anatomically correct finger and palm connections
- Red keypoint dots rendered at each of the 21 landmark positions
- Gesture recognition supporting 5 distinct gestures (Pinch, Fist, Open Hand, Peace, Thumbs Up)
- Real-time FPS counter displayed on the video feed
- Coordinate display showing landmark positions in pixel space
- GPU/CPU device selection with automatic detection and manual override
- YAML-based configuration system for all tunable parameters
- Graceful handling of 0, 1, or 2 hands in the camera frame without errors or visual artifacts

---

## Hardware Requirements

### Minimum

| Component       | Requirement                          |
