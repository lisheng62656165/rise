#!/usr/bin/env python3
"""Create a reproducible environment for the bundled OccuBench runner."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"


def python_in_venv() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def main() -> None:
    if not VENV.exists():
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV)])
    python = python_in_venv()
    if sys.version_info[:2] == (3, 12) and sys.platform.startswith("linux"):
        wheel_dir = ROOT / "vendor" / "wheels-linux-py312"
    elif sys.version_info[:2] == (3, 12) and os.name == "nt":
        wheel_dir = ROOT / "vendor" / "wheels-win-py312"
    else:
        wheel_dir = None
    if wheel_dir is not None and wheel_dir.exists():
        command = [str(python), "-m", "pip", "install", "--no-index", "--find-links", str(wheel_dir),
                   "-r", str(ROOT / "requirements.txt")]
    else:
        print("No bundled wheel set matches this platform; using the configured package index.",
              flush=True)
        command = [str(python), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")]
    subprocess.check_call(command)
    print(f"Environment ready: {python}")


if __name__ == "__main__":
    main()
