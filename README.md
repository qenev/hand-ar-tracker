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
