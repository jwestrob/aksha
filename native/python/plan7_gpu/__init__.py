"""Public GPU adapter API, loaded only when requested.

The CPU continuation/renderer extension can be installed without CUDA. Merely
importing this package (including ``from plan7_gpu import _pipeline``) must not
load ``adapter``, which requires the optional compiled GPU extension.
"""

from importlib import import_module as _import_module

__all__ = [
    "CandidateBatch",
    "PressedProfilePair",
    "ProfileSelection",
    "ProfileSession",
    "SequenceBatch",
    "cpu_candidates",
    "filter_ssv",
    "load_pressed_profiles",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import_module(".adapter", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
