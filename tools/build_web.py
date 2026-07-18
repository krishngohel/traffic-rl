"""Stage and build the browser version of the viewer.

Copies web/main.py plus the traffic_rl package (source + trained weights,
minus caches) into web_build/, then runs pygbag to compile the WASM bundle.
The deployable static site ends up in web_build/build/web/ — host it anywhere
(Netlify drag-and-drop, GitHub Pages, any static server). Serve locally for
testing with:  python -m pygbag --port 8000 web_build
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGE = ROOT / "web_build"


def stage() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir()
    shutil.copy2(ROOT / "web" / "main.py", STAGE / "main.py")
    shutil.copytree(
        ROOT / "src" / "traffic_rl",
        STAGE / "traffic_rl",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    print(f"staged -> {STAGE}")


def build(serve: bool = False) -> int:
    args = [sys.executable, "-m", "pygbag"]
    if not serve:
        args.append("--build")
    args.append(str(STAGE))
    print("running:", " ".join(args))
    return subprocess.call(args)


if __name__ == "__main__":
    stage()
    raise SystemExit(build(serve="--serve" in sys.argv))
