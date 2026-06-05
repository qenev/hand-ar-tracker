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


