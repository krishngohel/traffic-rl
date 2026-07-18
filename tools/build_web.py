"""Stage and build the browser version of the viewer.

Copies web/main.py plus the traffic_rl package (source + trained weights,
minus caches) into web_build/, then runs pygbag to compile the WASM bundle.
The deployable static site ends up in web_build/build/web/ — host it on any
REAL domain (Netlify drag-and-drop, GitHub Pages, ...): the runtime then
pulls numpy/pygame wheels from the public pygame-web CDN.

LOCAL TESTING MUST USE PYGBAG'S OWN SERVER on the default port:
    python -m pygbag --ume_block 0 web_build     # then open localhost:8000
When the page is served from localhost, the runtime enters dev mode and
expects wheels at the local /cdn/ path, which only pygbag's server provides —
a plain static file server will die with ModuleNotFoundError: numpy.
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
    # ume_block 0: boot straight into the animation, no click-to-start gate
    # (we play no audio, so no autoplay policy applies).
    args = [sys.executable, "-m", "pygbag", "--ume_block", "0"]
    if not serve:
        args.append("--build")
    args.append(str(STAGE))
    print("running:", " ".join(args))
    return subprocess.call(args)


if __name__ == "__main__":
    stage()
    raise SystemExit(build(serve="--serve" in sys.argv))
