"""AdaptiveAV package.

Expose convenient symbols for users of the library.  Most code lives in
individual modules (engine, adaptive, model, etc.) but importing this package
allows access to the core classes without knowing the module paths.
"""

from .engine import AdaptiveAVEngine, C, cprint
from .adaptive import TrulyAdaptiveClassifier, FEATURE_NAMES
from .model import TransformerAVModel

__all__ = [
    "AdaptiveAVEngine", "C", "cprint",
    "TrulyAdaptiveClassifier", "FEATURE_NAMES",
    "TransformerAVModel",
]
