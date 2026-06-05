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
