"""Process-parallel execution of independent seeded runs.

Every experiment in this project is a bag of independent (controller, config,
seed) episodes — paired seeds only require that run k use the same seed for
every controller, not that runs execute in any order. So distributing runs
across worker processes is statistically invisible: the same seeds produce the
same trajectories, bit for bit, and result order is preserved by map().

Controllers cross the process boundary as picklable *specs* (registry name,
fixed-time plan, ...) and are constructed inside the worker — the same
fresh-construction-per-run semantics the serial harness already uses.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor

# Controller spec forms, resolved in the worker process:
#   ("registry", name)           -> CONTROLLER_REGISTRY[name]()
#   ("fixed", plan)              -> FixedTimeController(plan)
#   ("scheduled", plans)         -> ScheduledFixedTimeController(plans)
#   ("pattern_weights", path)    -> PatternRLController(weights=path)
#   ("network", name)            -> network_controller_registry()[name]()


def resolve_jobs(jobs: int | None, n_tasks: int) -> int:
    if jobs is None or jobs <= 0:
        jobs = os.cpu_count() or 1
    return max(1, min(jobs, n_tasks))


def _limit_worker_threads() -> None:
    # The sim's numpy arrays are tiny; per-process BLAS threading only adds
    # contention when every core already runs its own worker.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")


def build_controller(spec: tuple):
    kind, payload = spec
    if kind == "registry":
        from traffic_rl.controllers import CONTROLLER_REGISTRY

        return CONTROLLER_REGISTRY[payload]()
    if kind == "fixed":
        from traffic_rl.controllers.fixed_time import FixedTimeController

        return FixedTimeController(payload)
    if kind == "scheduled":
        from traffic_rl.controllers.fixed_time import ScheduledFixedTimeController

        return ScheduledFixedTimeController(payload)
    if kind == "pattern_weights":
        from traffic_rl.rl.pattern_policy import PatternRLController

        return PatternRLController(weights=payload)
    if kind == "network":
        from traffic_rl.eval.network_harness import network_controller_registry

        return network_controller_registry()[payload]()
    if kind == "coordinated":
        from traffic_rl.controllers.network import ScheduledCoordinatedController

        return ScheduledCoordinatedController(payload)
    raise ValueError(f"unknown controller spec kind {kind!r}")


def run_single_task(task: tuple) -> dict:
    """(spec, config, seed) -> single-intersection run metrics."""
    from traffic_rl.eval.harness import run_controller

    spec, config, seed = task
    return run_controller(build_controller(spec), config, seed)


def run_network_task(task: tuple) -> dict:
    """(spec, config, seed) -> corridor run metrics."""
    from traffic_rl.eval.network_harness import run_network_controller

    spec, config, seed = task
    return run_network_controller(build_controller(spec), config, seed)


class RunPool:
    """A process pool reused for the whole experiment (Windows spawn makes
    worker startup expensive; pay it once, not per batch). jobs=1 runs
    everything inline for debugging and exact backwards compatibility."""

    def __init__(self, jobs: int | None = None):
        self.jobs = jobs if jobs and jobs > 0 else (os.cpu_count() or 1)
        self._executor: ProcessPoolExecutor | None = None

    def __enter__(self) -> RunPool:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None

    def map(self, fn: Callable[[tuple], dict], tasks: Iterable[tuple]) -> list[dict]:
        tasks = list(tasks)
        if self.jobs == 1 or len(tasks) <= 1:
            return [fn(t) for t in tasks]
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=min(self.jobs, os.cpu_count() or 1),
                initializer=_limit_worker_threads,
            )
        return list(self._executor.map(fn, tasks))
