"""Pytest configuration: make `baseline/` importable as a top-level package."""

import os
import sys
from pathlib import Path

# Add baseline/ to sys.path so `import src.X` works regardless of pytest's CWD.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
