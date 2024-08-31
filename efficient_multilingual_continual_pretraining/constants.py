"""Constants for the project."""

from pathlib import Path


_current_file_path = Path(__file__).resolve()
PROJECT_ROOT = _current_file_path.parent.parent

__all__ = [
    "PROJECT_ROOT",
]
