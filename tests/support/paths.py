"""Repository paths shared by tests regardless of their domain directory."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
