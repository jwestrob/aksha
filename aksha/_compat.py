"""Version compatibility helpers for Aksha."""

from __future__ import annotations

import sys
from importlib import import_module


def load_tomllib():
    """Load tomllib (stdlib in 3.11+) with fallback to tomli."""
    if sys.version_info >= (3, 11):
        return import_module("tomllib")
    return import_module("tomli")
