"""Compute device detection and selection utilities.

Provides functions to detect available compute devices (CPU, CUDA, MPS),
present an interactive selection prompt at startup, and manage device
fallback logic. The user can choose which discrete GPU to use.
"""

import sys
import subprocess
from typing import List, Dict, Optional

try:
    import torch
except ImportError:
    torch = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_available_devices() -> List[Dict[str, str]]:
    """Detect and list all available compute devices on the system.

    Queries PyTorch (for CUDA) and falls back to nvidia-smi for GPU
    name resolution. CPU is always included as option 0.

    Returns:
        A list of dicts, each with keys: index (int), type (str), name (str).
    """
    devices: List[Dict[str, str]] = []
    devices.append({"index": 0, "type": "cpu", "name": "CPU (no GPU acceleration)"})
    devices = _add_cuda_devices(devices)
    devices = _add_mps_device(devices)
    return devices


def select_device(config_preference: str) -> "torch.device":
    """Select a compute device based on the config preference.

    Modes:
    - 'auto'     → Interactive numbered list shown at startup.
    - 'cuda:0'   → Directly use that CUDA device (with availability check).
    - 'cpu'      → Force CPU.

    Args:
        config_preference: Value from config.yaml under the 'device' key.

    Returns:
        A torch.device (or CPU-compatible SimpleNamespace if torch absent).
    """
    if torch is None:
        print("[WARNING] PyTorch not installed – defaulting to CPU.")
        return _create_cpu_device()

    preference = config_preference.strip().lower()

    if preference == "auto":
        return _interactive_device_selection()
    return _direct_device_selection(preference)


def get_device_label(device) -> str:
    """Generate a human-readable on-screen label for the active device.

    Args:
        device: A torch.device or SimpleNamespace with a .type attribute.

    Returns:
        E.g. 'GPU: RTX 3050 Laptop (CUDA:0)' or 'Device: CPU'.
    """
    if torch is None or not hasattr(device, "type"):
        return "Device: CPU"

    dtype = str(device.type).lower()

    if dtype.startswith("cuda"):
        idx = device.index if device.index is not None else 0
        try:
            name = torch.cuda.get_device_name(idx)
            return f"GPU: {name} (CUDA:{idx})"
        except Exception:
            return f"GPU: CUDA:{idx}"

    if dtype == "mps":
        return "GPU: Apple Silicon (MPS)"

    return "Device: CPU"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _add_cuda_devices(devices: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Append all available NVIDIA CUDA devices to the list."""
    if torch is None:
        return devices
    if not torch.cuda.is_available():
        return devices
    for i in range(torch.cuda.device_count()):
        try:
            name = torch.cuda.get_device_name(i)
        except Exception:
            name = f"CUDA device {i}"
        devices.append({
            "index": len(devices),
            "type": f"cuda:{i}",
            "name": f"NVIDIA {name} (CUDA:{i})",
        })
    return devices


def _add_mps_device(devices: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Append Apple Silicon MPS device if available."""
    if torch is None:
        return devices
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append({
            "index": len(devices),
            "type": "mps",
            "name": "Apple Silicon GPU (MPS)",
        })
    return devices


def _interactive_device_selection():
    """Print available devices and prompt the user to pick one."""
    devices = list_available_devices()

    print("\n+----------------------------------------------+")
    print("|    Hand AR Tracker -- GPU Selection          |")
    print("+----------------------------------------------+")
    for dev in devices:
        marker = "   "
        if dev["type"].startswith("cuda"):
            marker = "[*]"  # highlight GPUs
        line = f"| {marker} [{dev['index']}] {dev['name']}"
        print(f"{line:<48}|")
    print("+----------------------------------------------+")

    # Default: first CUDA device, or 0 (CPU)
    default_idx = 0
    for dev in devices:
        if dev["type"].startswith("cuda"):
            default_idx = dev["index"]
            break

    try:
        raw = input(f"Select device number [{default_idx}]: ").strip()
        chosen = int(raw) if raw else default_idx
        if 0 <= chosen < len(devices):
            selected = devices[chosen]
            print(f"[INFO] Using: {selected['name']}\n")
            return torch.device(selected["type"])
        else:
            print(f"[WARNING] Invalid choice, using default [{default_idx}].\n")
            selected = devices[default_idx]
            return torch.device(selected["type"])
    except (ValueError, KeyboardInterrupt):
        print(f"\n[WARNING] No valid input -- using default [{default_idx}].\n")
        return torch.device(devices[default_idx]["type"])


def _direct_device_selection(preference: str):
    """Use a device specified directly in config (e.g. 'cuda:0')."""
    if preference == "cpu":
        print("[INFO] Config set to CPU.")
        return _create_cpu_device()

    if _is_cuda_available(preference):
        print(f"[INFO] Config device: {preference.upper()}")
        return torch.device(preference)

    print(f"[WARNING] Device '{preference}' not available – falling back to CPU.")
    return _create_cpu_device()


def _is_cuda_available(device_type: str) -> bool:
    """Check whether a given CUDA device string is usable."""
    if torch is None:
        return False
    if device_type.startswith("cuda"):
        if not torch.cuda.is_available():
            return False
        parts = device_type.split(":")
        if len(parts) == 2:
            try:
                return int(parts[1]) < torch.cuda.device_count()
            except ValueError:
                return False
        return True
    if device_type == "mps":
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    return False


def _create_cpu_device():
    """Return a CPU device (uses SimpleNamespace when torch is absent)."""
    if torch is not None:
        return torch.device("cpu")
    from types import SimpleNamespace
    return SimpleNamespace(type="cpu", index=None)
