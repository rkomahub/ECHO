import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def software_provenance() -> dict:
    """Report installed versions and Git state for the ECHO source checkout."""
    packages = {}
    for name in ("echo-spin", "numpy", "scipy", "qutip", "matplotlib"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None

    report = {
        "python": platform.python_version(),
        "packages": packages,
        "git_revision": None,
        "git_dirty": None,
    }

    root = Path(__file__).resolve().parents[2]
    if not (root / "pyproject.toml").is_file():
        return report
    if not (root / ".git").exists():
        return report

    try:
        revision = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return report

    report["git_revision"] = revision.stdout.strip()
    report["git_dirty"] = bool(status.stdout.strip())
    return report
