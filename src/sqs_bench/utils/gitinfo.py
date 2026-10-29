"""Optional git provenance for the run record (silently skipped outside a repo)."""
from __future__ import annotations

import subprocess
from pathlib import Path


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def git_info(cwd: str | Path | None = None) -> dict[str, str | bool | None]:
    root = Path(cwd) if cwd is not None else Path(__file__).resolve().parents[3]
    sha = _git(["rev-parse", "HEAD"], root)
    if sha is None:
        return {"commit": None, "dirty": None, "branch": None}
    status = _git(["status", "--porcelain"], root)
    return {
        "commit": sha,
        "dirty": bool(status),
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], root),
    }
