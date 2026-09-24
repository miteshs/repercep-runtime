"""Worktree-local pytest config: ensure tests import this worktree's source.

The shared `.venv` carries an editable install of `repercep-runtime` pointed at
whichever worktree happened to run `make install` first.  Without this hook,
``pytest`` from a parallel worktree would import the *other* worktree's
`src/repercep/`.  Insert this worktree's `src/` ahead on `sys.path` so any
import resolves locally.
"""

from __future__ import annotations

import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
_src = _here / "src"
if _src.is_dir() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
