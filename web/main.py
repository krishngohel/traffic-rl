"""traffic-rl in the browser: the full simulator + viewer compiled to
WebAssembly with pygbag. Everything runs client-side on the visitor's CPU —
no server, no data leaves the page. The settings button in the window switches
controllers (including the self-learning AI), scenarios, speed, and cameras.

Build (from the repo root):
    python tools/build_web.py
which stages this file plus the traffic_rl package into web_build/ and runs
pygbag; the deployable site lands in web_build/build/web/.
"""

# /// script
# dependencies = [
#   "numpy",
#   "pygame-ce",
# ]
# ///

import asyncio

import numpy  # noqa: F401  (declared so pygbag stages the wasm wheel)

from traffic_rl.viewer.app import ViewerApp


async def main() -> None:
    # Web default: the browser CPU is slower than a desktop, so start at 4x.
    app = ViewerApp("actuated", "arterial_lefts", speed=4.0, seed=42)
    await app.run_async()


asyncio.run(main())
