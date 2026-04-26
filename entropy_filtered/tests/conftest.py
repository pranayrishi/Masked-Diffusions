"""Pytest configuration for entropy_filtered/.

We add the *repo root* to sys.path so both `from baseline.src.X import Y`
and `from entropy_filtered.src.X import Y` resolve, regardless of pytest's
working directory or rootdir.
"""

import sys
from pathlib import Path

# entropy_filtered/tests/conftest.py → entropy_filtered/tests → entropy_filtered → REPO_ROOT
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
