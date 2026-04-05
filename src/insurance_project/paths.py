from __future__ import annotations

from pathlib import Path


def find_project_root(start: str | Path | None = None) -> Path:
    """
    Find the project root by walking upward until we see `data/` and `notebooks/`.
    Falls back to the provided start directory if nothing is found.
    """
    current = Path(start) if start is not None else Path.cwd()
    current = current.resolve()

    for candidate in [current, *current.parents]:
        if (candidate / "data").exists() and (candidate / "notebooks").exists():
            return candidate

    return current


def data_path(filename: str, *, start: str | Path | None = None) -> Path:
    return find_project_root(start) / "data" / filename


def models_path(filename: str, *, start: str | Path | None = None) -> Path:
    return find_project_root(start) / "models" / filename

