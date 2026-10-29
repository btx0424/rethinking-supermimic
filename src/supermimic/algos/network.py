"""Re-export ``supermimic.supermimic.network`` as ``supermimic.algos.network``."""

import importlib
import sys

_mod = importlib.import_module("supermimic.supermimic.network")
sys.modules[__name__] = _mod
