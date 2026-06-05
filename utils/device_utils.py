"""Compute device detection and selection utilities.

Provides functions to detect available compute devices (CPU, CUDA, MPS),
present a selection prompt to the user, and manage device fallback logic.
Integrates with PyTorch for device management.
"""

import sys
from typing import List, Dict, Optional

try:
    import torch
