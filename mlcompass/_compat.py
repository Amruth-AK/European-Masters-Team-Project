"""
mlcompass._compat — Logging wrapper & progress callback protocol
================================================================
"""

import logging
from typing import Callable, Optional, Protocol, runtime_checkable

# Shared logger for all mlcompass modules
logger = logging.getLogger("mlcompass")


@runtime_checkable
class ProgressCallback(Protocol):
    """Protocol for progress reporters.

    Any callable accepting a single float (0.0–1.0) satisfies this protocol.
    Streamlit progress bars, tqdm updaters, or plain functions all work.
    """
    def __call__(self, fraction: float) -> None: ...
