# Hand AR Tracker

*Real-time hand AR system with ASCII art effects, holographic ribbon, and gesture controls — built with Python + MediaPipe.*

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Hardware Requirements](#hardware-requirements)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Gesture Controls](#gesture-controls)
7. [Keyboard Controls](#keyboard-controls)
8. [Effect Modes](#effect-modes)
9. [Configuration](#configuration)
10. [Project Architecture](#project-architecture)
11. [License](#license)

---

## Overview

Hand AR Tracker captures your webcam feed and overlays real-time augmented reality effects driven entirely by **hand gestures**, **finger snaps**, and **eye winks** — no buttons, no keyboard, just your body.

The core effect is a **holographic ASCII ribbon** that stretches between your two hands, rendering your silhouette as scrolling ASCII scanlines against a black background. A **snap gesture** inverts the entire effect, and a **double wink** locks the ribbon's position in 3D space.

---

## Features

- 🖐 **Dual-hand tracking** — up to two hands simultaneously via MediaPipe Hands
- ✨ **Holographic ASCII ribbon** — scanline particle effect renders your silhouette as ASCII art between both hands
- 🔄 **Snap-to-reverse** — snap your fingers to flip the effect: real video inside ribbon, ASCII art outside
- 😉 **Double-wink lock** — wink twice in one second to freeze the ribbon position in space
- 🎯 **Person + object segmentation** — MediaPipe Selfie Segmentation + background subtraction
- 📟 **Live FPS counter** with rolling average
- 🖱 **Click-to-flip** camera rotation button in the HUD
- ⚡ Auto GPU/CPU selection (CUDA or CPU)
- 🎨 Fully configurable via `config.yaml`

---

## Hardware Requirements

### Minimum

| Component | Requirement |
|-----------|-------------|
| Camera    | Any USB/built-in webcam |
| Python    | 3.10+ |
| RAM       | 4 GB |
| OS        | Windows 10+, macOS 12+, Linux |

### Recommended

| Component | Recommendation |
|-----------|----------------|
| GPU       | NVIDIA GPU with CUDA |
| Camera    | 1080p webcam at 60 fps |
| RAM       | 8 GB+ |
| CPU       | AMD Ryzen 5 / Intel i5 or better |

---

## Installation

### Windows (one-click)

1. **Clone the repo**
   ```bash
   git clone https://github.com/qenev/hand-ar-tracker.git
   cd hand-ar-tracker
   ```

2. **Run the installer** — creates the virtual environment and installs all dependencies:
   ```
   install.bat
   ```

3. **Launch the tracker:**
   ```
   start.bat
   ```

### Manual installation

```bash
git clone https://github.com/qenev/hand-ar-tracker.git
cd hand-ar-tracker

python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt

python main.py
```

---

## Quick Start

```bash
start.bat        # Windows one-click launch
# or
python main.py   # manual
```

The app opens your webcam. Show **both hands** to the camera to start.

---

## Gesture Controls

All effects are controlled entirely by **body gestures** — no keyboard required for the AR effects.

### 🤌 Double Pinch Touch — Toggle Ribbon ON/OFF

Bring both hands in front of the camera, **pinch** your thumb and index finger together on **each hand simultaneously**, then touch the two pinched fingertips together.

- **First double-pinch:** Ribbon turns **ON** (ASCII scanline effect between hands)
- **Second double-pinch:** Ribbon turns **OFF**
- If either hand leaves frame while the ribbon is off, the effect auto-resets

```
Both hands visible → pinch both → touch pinched tips together → toggle
```

### 👌 Snap — Reverse the Effect

Snap your fingers on either hand (bring your **thumb tip** and **middle fingertip** together then release quickly).

| Without Snap (default) | With Snap (reversed) |
|---|---|
| Ribbon ON → ASCII scanlines on your silhouette | Ribbon ON → Real video inside ribbon, ASCII art everywhere outside |
| Ribbon OFF → Normal camera | Ribbon OFF → Full-frame ASCII art on black background |

Snap again to revert back to normal mode.

### 😉 Double Wink — Lock / Unlock Ribbon Position

Wink **one eye** (not both — a full blink won't count) **twice within 1 second** to lock the ribbon at its current position in space. The ribbon will stay frozen even if you move your hands away.

- **Double-wink while ribbon is ON:** Position locks. You can lower your hands — the ribbon stays.
- **Double-wink while ribbon is locked:** Unlocks the ribbon so it follows your hands again.

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| **Q** | Quit the application |
| **ESC** | Toggle hand skeleton visibility (dots + lines) on/off |
| **Delete** | Toggle the entire HUD (FPS counter, flip button, device label) on/off |

### HUD — On-screen Controls

| Element | Location | Action |
|---------|----------|--------|
| **FPS counter** | Top-left | Live rolling-average frame rate |
| **Flip 180 button** | Top-center (clickable) | Click to rotate camera 180° for upside-down mounts |
| **Device label** | Top-right | Shows `CUDA` or `CPU` |

---

## Effect Modes

### Default Mode (no snap)

```
Ribbon OFF:  Normal camera feed
Ribbon ON:   Your silhouette rendered as scrolling ASCII scanlines
             between your two hands, against a black background.
             Rest of frame is plain black.
```

### Reversed Mode (after snap)

```
Ribbon OFF:  Full frame converted to ASCII art on black background
Ribbon ON:   Inside ribbon = clear real video
             Outside ribbon = ASCII art
             (No border — clean hard edge)
```

### Locked Ribbon (after double wink)

The ribbon polygon stays fixed in screen-space at the position captured when the wink fired. Works in both default and reversed modes.

---

## Configuration

All settings live in `config.yaml`.

### Camera

```yaml
camera:
  index: 0          # webcam index (0 = default)
  width: 1280
  height: 720
  fps: 60
  rotate_180: false  # true if camera is mounted upside-down
```

### Tracking

```yaml
tracking:
  max_hands: 2
  min_detection_confidence: 0.8
  min_tracking_confidence: 0.7
  model_complexity: 1   # 0 = lite/fast, 1 = full/accurate
```

### Renderer

```yaml
renderer:
  skeleton_color: [0, 255, 0]     # BGR — green
  keypoint_color: [0, 0, 255]     # BGR — red
  keypoint_radius: 6
  skeleton_thickness: 2
  show_fps: true
  show_coordinates: false
  show_gesture_label: false
  show_hand_label: false
  show_device_label: true
```

### Gestures

```yaml
gestures:
  enabled: false         # enable named gesture labels (Pinch, Fist, etc.)
  pinch_threshold: 0.05  # thumb-to-index distance to count as a pinch
  fist_threshold: 0.85
```

### Device

```yaml
device: auto   # "auto" | "cuda:0" | "cpu"
```

---

## Project Architecture

| File | Purpose |
|------|---------|
| `main.py` | Entry point, main loop, wink detection, snap detection, mode routing |
| `tracker.py` | MediaPipe Hands wrapper + FaceMesh + SelfieSegmentation |
| `renderer.py` | All OpenCV drawing: skeleton, ribbon (ASCII + real), HUD |
| `ascii_processor.py` | Vectorised ASCII art renderer (CLAHE + char tiles) |
| `gesture.py` | Per-hand gesture classification |
| `config.yaml` | All runtime configuration |
| `utils/fps_counter.py` | Rolling-window FPS counter |
| `utils/device_utils.py` | GPU/CPU auto-detection |
| `utils/math_utils.py` | Landmark smoothing and coordinate helpers |
| `install.bat` | One-click Windows dependency installer |
| `start.bat` | One-click Windows launcher |
| `assets/` | Reference images and ASCII character data |

---

## License

MIT License — see [LICENSE](LICENSE) for details.
