"""Safety invariants asserted from the event log, controller-independently."""

import numpy as np
import pytest

from conftest import AdversarialController, StubbornController, drive
from traffic_rl.controllers import CONTROLLER_REGISTRY
from traffic_rl.sim.core import IntersectionSim
from traffic_rl.sim.signal import SignalState

TWO_HOURS = 7200

GREEN, YELLOW, ALL_RED = (int(s) for s in SignalState)


def _runs(states: np.ndarray, phases: np.ndarray):
    """Consecutive (state, phase, length) runs from per-step logs."""
    out = []
    start = 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[start] or phases[i] != phases[start]:
            out.append((int(states[start]), int(phases[start]), i - start))
            start = i
    return out


def _controllers():
    yield AdversarialController()
    yield StubbornController()
    for factory in CONTROLLER_REGISTRY.values():
        yield factory()


@pytest.mark.parametrize("controller", list(_controllers()), ids=lambda c: c.name)
def test_invariants(busy_config, controller):
    sim = IntersectionSim(busy_config)
    log = drive(sim, controller, seed=123, n_steps=TWO_HOURS)
    states, phases = log["step_state"], log["step_phase"]
    runs = _runs(states, phases)
    timing = busy_config.timing

    for i, (state, phase, length) in enumerate(runs):
        final = i + 1 == len(runs)  # horizon may truncate the last run mid-state
        if state == YELLOW:
            assert runs[i - 1][0] == GREEN, "yellow must follow green"
            if not final:
                assert length * busy_config.dt >= timing.yellow, "yellow shorter than 3 s"
                assert runs[i + 1][0] == ALL_RED, "all-red must follow yellow"
        if state == ALL_RED and not final:
            assert length * busy_config.dt >= timing.all_red, "all-red shorter than 2 s"
            nxt = runs[i + 1]
            assert nxt[0] == GREEN
            assert nxt[1] != phase, "phase must change across a transition"
        if state == GREEN and not final:  # completed greens only
            assert length * busy_config.dt >= timing.min_green, "min green violated"

    # Walk + clearance never truncated: after each true walk start on movement m,
    # the signal stays green on phase m for the full ped service time.
    ws, pa, pm = log["ped_walk_start"], log["ped_arrival"], log["ped_movement"]
    served = ~np.isnan(ws) & (ws > pa)  # peds who waited, then were served by a walk
    walk_starts = {(float(w), int(m)) for w, m in zip(ws[served], pm[served], strict=True)}
    t = log["step_t"]
    horizon = t[-1]
    # Logged state at step t applies to [t, t+dt): a walk starting at w must keep
    # the signal green on phase m for all steps in [w, w + ped_service).
    for w, m in walk_starts:
        if w + timing.ped_service > horizon:
            continue
        window = (t >= w) & (t < w + timing.ped_service)
        assert (states[window] == GREEN).all(), "walk truncated: not green throughout"
        assert (phases[window] == m).all(), "walk truncated: phase changed"


def test_backstop_forces_switch(busy_config):
    """A stubborn controller cannot starve the conflicting approaches forever."""
    sim = IntersectionSim(busy_config)
    log = drive(sim, StubbornController(), seed=5, n_steps=TWO_HOURS)
    runs = _runs(log["step_state"], log["step_phase"])
    green_lengths = [length for state, _, length in runs if state == GREEN]
    # Backstop 120 s, plus at most one ped service (20 s) of lock overrun.
    limit = busy_config.timing.max_green_backstop + busy_config.timing.ped_service
    assert max(green_lengths) * busy_config.dt <= limit + 1.0
    assert len(green_lengths) > 50, "backstop should force regular switching"
