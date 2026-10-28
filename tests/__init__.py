"""Project regression tests, separate from the installed runtime packages."""

from pathlib import Path
import os
import sys


_ROOT = Path(__file__).resolve().parents[1]
_PATHS = [str(_ROOT / "src"), str(_ROOT)]
for _path in reversed(_PATHS):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Subprocess regressions must exercise the same checkout as the parent.
_inherited = [path for path in os.environ.get("PYTHONPATH", "").split(os.pathsep) if path]
os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([*_PATHS, *_inherited]))
