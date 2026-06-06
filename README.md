# Hand AR Tracker

*Real-time hand and finger tracking augmented reality system built with Python.*

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Hardware Requirements](#hardware-requirements)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Configuration](#configuration)
7. [Running the Application](#running-the-application)
8. [Gestures Reference](#gestures-reference)
9. [Project Architecture](#project-architecture)
10. [Running Tests](#running-tests)
11. [Contributing](#contributing)
12. [License](#license)

---

## Overview

Hand AR Tracker captures video from a standard webcam, detects and tracks both hands simultaneously using **Google MediaPipe**, and renders a **green skeletal mesh** connecting all 21 hand landmarks per hand. Each landmark is highlighted with a **red dot**. The system also includes optional **gesture recognition** for five common gestures and displays a live FPS counter and optional landmark coordinates.

---

## Features

- Dual‑hand tracking with up to two hands simultaneously.
- 21 landmarks per hand, following the MediaPipe hand model.
- Green skeletal mesh overlay and red keypoint dots.
- Real‑time FPS counter and optional coordinate display.
- Configurable via `config.yaml` (device selection, confidence thresholds, colors, display options, etc.).
- Optional gesture recognition: Pinch, Fist, Open Hand, Peace, Thumbs Up.
- Graceful handling of 0, 1, or 2 hands in view.
- Supports CPU and CUDA‑enabled GPU execution.

---

## Hardware Requirements

### Minimum

| Component | Requirement |
|-----------|-------------|
| Camera    | Any USB webcam |
| Python    | 3.10+ |
| RAM       | 4 GB |
| CPU       | Modern x86_64 or ARM |
| OS        | Windows 10+, macOS 12+, Linux |

### Recommended

| Component | Recommendation |
|-----------|-----------------|
| GPU       | NVIDIA GPU with CUDA |
| Camera    | 1080p webcam |
| RAM       | 8 GB+ |
| CPU       | Intel i5 / AMD Ryzen 5 or better |

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/hand-ar-tracker.git
   cd hand-ar-tracker
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   ```

3. **Activate the environment**

   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS / Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

5. **Verify installation**

   ```bash
   python -c "import cv2, mediapipe, torch; print('Dependencies OK')"
   ```

---

## Quick Start

Run the demo with a single command:

```bash
python main.py
```

The application will:

1. Load `config.yaml` (or use defaults).
2. Detect GPU/CPU based on the `device` setting.
3. Open the webcam feed and overlay hand landmarks.
4. Display FPS, optional coordinates, and recognized gestures.

Press **q** or **ESC** to exit gracefully.

---

## Configuration

All tunable parameters live in `config.yaml`. Below is the full reference:

### Camera Settings (`camera`)
- `index` (integer, default `0`): Webcam index.
- `width` (integer, default `1280`): Capture width in pixels.
- `height` (integer, default `720`): Capture height in pixels.
- `fps` (integer, default `60`): Target frame rate.
- `rotate_180` (boolean, default `true`): Flip the camera 180 degrees (upside down).

### Tracking Settings (`tracking`)
- `max_hands` (integer, default `2`): Max hands to detect and track (1 or 2).
- `min_detection_confidence` (float, default `0.8`): Confidence threshold for initial detection.
- `min_tracking_confidence` (float, default `0.7`): Confidence threshold for frame-to-frame tracking.
- `model_complexity` (integer, default `1`): MediaPipe model complexity (0 = lite, 1 = full/accurate).

### Renderer Settings (`renderer`)
- `skeleton_color` (list, default `[0, 255, 0]`): BGR color for skeleton lines (green).
- `keypoint_color` (list, default `[0, 0, 255]`): BGR color for landmark dots (red).
- `keypoint_radius` (integer, default `4`): Radius of landmark dots.
- `skeleton_thickness` (integer, default `2`): Thickness of skeleton lines.
- `show_fps` (boolean, default `true`): Display FPS counter.
- `show_coordinates` (boolean, default `false`): Show coordinates for finger tips.
- `show_gesture_label` (boolean, default `false`): Show detected gesture names.
- `show_hand_label` (boolean, default `false`): Show Left/Right hand labels.
- `show_device_label` (boolean, default `true`): Show active compute device.

### Gestures Settings (`gestures`)
- `enabled` (boolean, default `false`): Enable/disable gesture recognition.
- `pinch_threshold` (float, default `0.05`): Distance threshold for pinching.
- `fist_threshold` (float, default `0.85`): Ratio threshold for fist gesture.

### General Settings
- `device` (string, default `"cuda:0"`): Active device. Use `"auto"` to choose at startup, `"cuda:0"` to force Nvidia GPU, or `"cpu"`.

---

## Running the Application

```bash
python main.py
```

### Command‑line Options

| Option | Description |
|--------|-------------|
| `--config PATH` | Path to an alternative configuration file. |
| `--no-gpu`      | Force CPU execution even if a GPU is available. |
| `--debug`       | Enable verbose debug logging to the console. |

---

## Gestures Reference

| Gesture | Description |
|---------|-------------|
| **Pinch** | Thumb tip touches index fingertip; other fingers extended. |
| **Fist** | All fingers curled into the palm. |
| **Open Hand** | All fingers fully extended, palm facing camera. |
| **Peace** | Index and middle fingers extended; others curled. |
| **Thumbs Up** | Thumb extended upward; remaining fingers curled. |

When two hands are present, each hand reports its own gesture independently.

---

## Project Architecture

| Path | Purpose |
|------|---------|
| `main.py` | Entry point, sets up components and main loop. |
| `config.yaml` | Default configuration values. |
| `requirements.txt` | Python package dependencies. |
| `tracker.py` | MediaPipe hand detection and landmark extraction. |
| `gesture.py` | Gesture classification logic. |
| `renderer.py` | Rendering of landmarks, connections, HUD elements. |
| `utils/` | Helper utilities (FPS calculation, etc.). |
| `tests/` | Unit tests for all modules. |
| `assets/` | Demo screenshots and reference images. |

---

## Running Tests

```bash
pytest tests/
```

All tests should pass. For verbose output:

```bash
pytest -v tests/
```

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/my‑feature`).
3. Make your changes, ensuring code style consistency (PEP 8).
4. Add or update tests as needed.
5. Run the full test suite.
6. Submit a Pull Request with a clear description of changes.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

