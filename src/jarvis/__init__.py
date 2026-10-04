"""Jarvis: a persistent personal assistant core.

The language model is one component of the system, never its authority: policy, permissions,
state and verification are deterministic and owned by Jarvis.
"""

from jarvis.bootstrap import build_jarvis
from jarvis.config import Settings
from jarvis.orchestration.jarvis import Jarvis

__version__ = "0.6.0"

__all__ = ["Jarvis", "Settings", "__version__", "build_jarvis"]
