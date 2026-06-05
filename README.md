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
|-----------------|--------------------------------------|
| Camera          | Any USB webcam (built-in or external)|
| Python          | 3.10 or higher                       |
| RAM             | 4 GB                                 |
| CPU             | Any modern x86_64 or ARM processor   |
| Operating System| Windows 10+, macOS 12+, or Linux     |

### Recommended

| Component       | Recommendation                              |
|-----------------|---------------------------------------------|
| GPU             | Dedicated NVIDIA GPU with CUDA support      |
| Camera          | 1080p webcam for higher fidelity tracking   |
| RAM             | 8 GB or more                                |
| CPU             | Intel i5 / AMD Ryzen 5 or better            |

---

## Installation

Follow these steps to set up the project from scratch:

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/hand-ar-tracker.git
   cd hand-ar-tracker
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   ```

   Activate the virtual environment:

   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS / Linux:
     ```bash
     source venv/bin/activate
     ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**

   ```bash
   python -c "import cv2; import mediapipe; import torch; print('All dependencies installed successfully.')"
   ```

   If no errors are printed, the environment is ready.

---

## How to Run

Start the application with the following command:

```bash
python main.py
```

### What happens on startup

1. The configuration file `config.yaml` is loaded. If no file is found, default values are used.
2. The device selection system determines whether to use GPU or CPU for torch operations.
3. The webcam is initialized at the resolution and frame rate specified in the configuration.
4. A live video window opens showing the camera feed with hand tracking overlays.

### How to quit

Press **q** or **ESC** at any time while the video window is focused to gracefully shut down the application. The webcam and all OpenCV windows will be released and destroyed automatically.

---

## GPU Selection

The application includes a device selection system that determines whether PyTorch operations run on the GPU or CPU.

### Automatic mode

When `device` in `config.yaml` is set to `"auto"` (the default), the application checks for CUDA availability at startup. If a compatible GPU is detected, the user is prompted to confirm GPU usage. If declined, the system falls back to CPU.

### Configuration override

Setting `device` to `"cuda"` or `"cpu"` in `config.yaml` bypasses the interactive prompt entirely. The application will attempt to use the specified device directly.

- `"cuda"` -- Forces GPU usage. If no compatible GPU is found, the application falls back to CPU and logs a warning.
- `"cpu"` -- Forces CPU usage regardless of GPU availability.
- `"auto"` -- Prompts the user at startup if a GPU is detected.

### Example configuration

```yaml
device: "auto"   # Options: "auto", "cuda", "cpu"
```

---

## Configuration Reference

All configuration is managed through the `config.yaml` file in the project root. Below is a complete reference of every supported field.

| Field                     | Type    | Default   | Description                                                                 |
|---------------------------|---------|-----------|-----------------------------------------------------------------------------|
| `device`                  | string  | `"auto"`  | Device selection mode. Options: `"auto"`, `"cuda"`, `"cpu"`.                |
| `camera_index`            | integer | `0`       | Index of the webcam device to use. `0` is typically the default camera.     |
| `frame_width`             | integer | `1280`    | Capture width in pixels requested from the webcam.                          |
| `frame_height`            | integer | `720`     | Capture height in pixels requested from the webcam.                         |
| `fps`                     | integer | `30`      | Target frames per second for the webcam capture.                            |
| `max_hands`               | integer | `2`       | Maximum number of hands to detect simultaneously. Valid range: 1-2.         |
| `detection_confidence`    | float   | `0.7`     | Minimum confidence threshold for initial hand detection. Range: 0.0-1.0.   |
| `tracking_confidence`     | float   | `0.5`     | Minimum confidence threshold for hand landmark tracking. Range: 0.0-1.0.   |
| `draw_landmarks`          | boolean | `true`    | Whether to draw red keypoint dots on detected landmarks.                    |
| `draw_connections`        | boolean | `true`    | Whether to draw green skeletal mesh connections between landmarks.           |
| `show_fps`                | boolean | `true`    | Whether to display the FPS counter on the video feed.                       |
| `show_coordinates`        | boolean | `false`   | Whether to display pixel coordinates for each landmark.                     |
| `gesture_recognition`     | boolean | `true`    | Whether to enable gesture recognition and display detected gestures.        |
| `landmark_color`          | list    | `[0,0,255]` | BGR color for keypoint dots. Default is red.                              |
| `connection_color`        | list    | `[0,255,0]` | BGR color for skeletal mesh lines. Default is green.                      |
| `landmark_radius`         | integer | `5`       | Radius in pixels for each keypoint dot.                                     |
| `connection_thickness`    | integer | `2`       | Line thickness in pixels for skeletal mesh connections.                      |
| `flip_horizontal`         | boolean | `true`    | Whether to mirror the video feed horizontally for a natural mirror effect.  |

---

## Project Architecture

| File / Directory          | Purpose                                                                     |
|---------------------------|-----------------------------------------------------------------------------|
| `main.py`                 | Application entry point. Initializes components and runs the main loop.     |
| `config.yaml`             | YAML configuration file for all tunable parameters.                         |
| `requirements.txt`        | Python package dependencies.                                                |
| `hand_tracker.py`         | Core hand tracking module wrapping MediaPipe hand detection and landmarks.  |
| `gesture_recognizer.py`   | Gesture recognition logic for classifying hand poses into named gestures.   |
| `renderer.py`             | Rendering module for drawing landmarks, connections, and HUD elements.      |
| `device_selector.py`      | GPU/CPU device selection and validation logic.                               |
| `config_loader.py`        | YAML configuration loading and default value management.                    |
| `utils.py`                | Shared utility functions (FPS calculation, coordinate formatting, etc.).     |
| `tests/`                  | Directory containing unit tests for all modules.                            |
| `tests/test_gesture.py`   | Tests for gesture recognition accuracy and edge cases.                      |
| `tests/test_tracker.py`   | Tests for hand tracker initialization and landmark processing.              |
| `tests/test_config.py`    | Tests for configuration loading, defaults, and validation.                  |
| `assets/`                 | Static assets including demo screenshots and reference images.              |
| `LICENSE`                 | MIT License file.                                                           |
| `README.md`               | This file. Project documentation and usage guide.                           |

---

## Gesture Reference

The following five gestures are recognized when `gesture_recognition` is enabled in the configuration.

| Gesture        | How to Trigger                                                                                      |
|----------------|-----------------------------------------------------------------------------------------------------|
| Pinch          | Touch the tip of the thumb to the tip of the index finger while keeping other fingers extended.      |
| Fist           | Curl all five fingers into the palm so that no fingertips are extended.                              |
| Open Hand      | Extend all five fingers fully with the palm facing the camera.                                       |
| Peace          | Extend the index and middle fingers upward while curling the ring finger, pinky, and thumb.          |
| Thumbs Up      | Extend the thumb upward while curling all four remaining fingers into the palm.                      |

Gestures are evaluated per hand. When two hands are detected, each hand reports its own gesture independently.

---

## Running Tests

Run the full test suite with pytest:

```bash
pytest tests/
```

### Expected output

All tests should pass with output similar to:

```
========================= test session starts ==========================
collected 15 items

tests/test_config.py ....                                         [ 26%]
tests/test_gesture.py ......                                      [ 66%]
tests/test_tracker.py .....                                       [100%]

========================== 15 passed in 2.34s ==========================
```

To run tests with verbose output:

```bash
pytest tests/ -v
```

To run a specific test file:

```bash
pytest tests/test_gesture.py
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for full details.
