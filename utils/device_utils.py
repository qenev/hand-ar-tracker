"""Compute device detection and selection utilities.

Provides functions to detect available compute devices (CPU, CUDA, MPS),
present a selection prompt to the user, and manage device fallback logic.
Integrates with PyTorch for device management.
"""

import sys
from typing import List, Dict, Optional

try:
    import torch
except ImportError:
    torch = None


def list_available_devices() -> List[Dict[str, str]]:
    """Detect and list all available compute devices on the system.

    Queries PyTorch to find CPU, CUDA (NVIDIA GPU), and MPS (Apple
    Silicon GPU) devices. Each device is returned with an index,
    type identifier, and human-readable name.

    Returns:
        A list of dictionaries, each containing:
            - index: Integer index for device selection.
            - type: Device type string (cpu, cuda, mps).
            - name: Human-readable device name.
    """
    devices: List[Dict[str, str]] = []
    devices.append({
        "index": 0,
        "type": "cpu",
        "name": "CPU",
    })
    if torch is not None:
        devices = _add_cuda_devices(devices)
        devices = _add_mps_device(devices)
    return devices


def _add_cuda_devices(
    devices: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Add available CUDA GPU devices to the device list.

    Args:
        devices: Existing list of detected devices to append to.

    Returns:
        Updated device list with any CUDA devices appended.
    """
    if torch is not None and torch.cuda.is_available():
        cuda_count = torch.cuda.device_count()
        for i in range(cuda_count):
            device_name = torch.cuda.get_device_name(i)
            devices.append({
                "index": len(devices),
                "type": f"cuda:{i}",
                "name": f"CUDA:{i} -- {device_name}",
            })
    return devices


def _add_mps_device(
    devices: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Add Apple Silicon MPS device if available.

    Args:
        devices: Existing list of detected devices to append to.

    Returns:
        Updated device list with MPS device appended if available.
    """
    if torch is not None and hasattr(torch.backends, "mps"):
        if torch.backends.mps.is_available():
            devices.append({
                "index": len(devices),
                "type": "mps",
                "name": "MPS -- Apple Silicon GPU",
            })
    return devices


def select_device(config_preference: str) -> "torch.device":
    """Select a compute device based on config preference.

    Handles three modes:
    - 'auto': Displays available devices and prompts user to select one.
    - Specific device string (e.g. 'cpu', 'cuda:0', 'mps'): Attempts
      to use the specified device directly.
    - Falls back to CPU if the requested device is unavailable.

    Args:
        config_preference: Device preference string from config.yaml.
            Use 'auto' for interactive selection, or a device string
            like 'cpu', 'cuda:0', or 'mps' for direct selection.

    Returns:
        A torch.device object representing the selected compute device.
    """
    if torch is None:
        print("[WARNING] PyTorch not available, using CPU.")
        return _create_cpu_device()
    if config_preference.lower() == "auto":
        return _interactive_device_selection()
    return _direct_device_selection(config_preference)


def _interactive_device_selection() -> "torch.device":
    """Display available devices and prompt user for selection.

    Lists all detected devices with their indices and names, then
    waits for user input to select a device by number.

    Returns:
        The selected torch.device, or CPU device if selection fails.
    """
    devices = list_available_devices()
    print("\n--- Available Compute Devices ---")
    for device_info in devices:
        idx = device_info["index"]
        name = device_info["name"]
        print(f"  [{idx}] {name}")
    print("---------------------------------")
    return _prompt_device_selection(devices)


def _prompt_device_selection(
    devices: List[Dict[str, str]],
) -> "torch.device":
    """Prompt the user to select a device by index number.

    Args:
        devices: List of available device dictionaries with index,
            type, and name fields.

    Returns:
        The selected torch.device based on user input. Falls back
        to CPU on invalid input or errors.
    """
    try:
        choice = input("Select device by number [0]: ").strip()
        if choice == "":
            choice = "0"
        idx = int(choice)
        if 0 <= idx < len(devices):
            device_type = devices[idx]["type"]
            print(f"[INFO] Selected device: {devices[idx]['name']}")
            return torch.device(device_type)
    except (ValueError, IndexError, KeyboardInterrupt):
        pass
    print("[WARNING] Invalid selection, falling back to CPU.")
    return _create_cpu_device()


def _direct_device_selection(preference: str) -> "torch.device":
    """Attempt to select a specific device by type string.

    Args:
        preference: Device type string like 'cpu', 'cuda:0', or 'mps'.

    Returns:
        The requested torch.device if available, otherwise CPU
        with a warning message.
    """
    preference_lower = preference.lower().strip()
    if preference_lower == "cpu":
        print("[INFO] Using CPU as specified in config.")
        return _create_cpu_device()
    if _is_device_available(preference_lower):
        print(f"[INFO] Using device: {preference_lower}")
        return torch.device(preference_lower)
    print(
        f"[WARNING] Device '{preference}' not available. "
        f"Falling back to CPU."
    )
    return _create_cpu_device()

