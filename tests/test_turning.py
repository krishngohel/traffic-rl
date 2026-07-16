"""Physics of the Phase 5 movement model: protected lefts, permissive gap
acceptance, sneakers, shared-lane friction, and geometry-derived timings."""

import numpy as np
import pytest

from traffic_rl.config import (
    DemandConfig,
    GeometryConfig,
    IntersectionLayout,
    LeftTurnTreatment,
    SimConfig,
    build_phases,
    ite_all_red,
    ite_yellow,
    left_group,
    mutcd_ped_clearance,
    timing_from_geometry,
)
from traffic_rl.sim.core import IntersectionSim
from traffic_rl.sim.signal import SignalState

_S = LeftTurnTreatment.SHARED
_P = LeftTurnTreatment.PERMISSIVE
_X = LeftTurnTreatment.PROTECTED


def _run(config: SimConfig, controller, n_steps: int, seed: int = 11):
    sim = IntersectionSim(config)
    obs = sim.reset(seed)
    controller.reset(config, np.random.default_rng(seed))
    for _ in range(n_steps):
        obs = sim.step(controller.act(obs)).obs
    return sim


def _left_demand(treatments) -> SimConfig:
    return SimConfig(
        demand=DemandConfig(
            vehicle_rates=(500, 500, 400, 400),
            ped_rates=(20, 20),
            turn_fractions=((0.25, 0.65, 0.10),) * 4,
        ),
        layout=IntersectionLayout(left_turn=treatments),
    )


def test_phase_table_shapes():
    assert len(build_phases(IntersectionLayout())) == 2
    assert len(build_phases(IntersectionLayout(left_turn=(_X, _X, _P, _P)))) == 3
    assert len(build_phases(IntersectionLayout(left_turn=(_X, _X, _X, _X)))) == 4
    # Slots are canonical and ordered.
    slots = [p.slot for p in build_phases(IntersectionLayout(left_turn=(_X, _X, _X, _X)))]
    assert slots == [0, 1, 2, 3]


def test_protected_lefts_get_served():
    from traffic_rl.controllers.actuated import ActuatedController

    config = _left_demand((_X, _X, _X, _X))
    sim = _run(config, ActuatedController(), n_steps=3600)
    log = sim.event_log.finalize()
    groups = log["veh_group"]
    departed = ~np.isnan(log["veh_depart"])
    for a in range(4):
        left_arrivals = (groups == left_group(a)).sum()
        left_departed = (departed & (groups == left_group(a))).sum()
        assert left_arrivals > 50  # demand actually routed to the bay
        assert left_departed > 0.8 * left_arrivals - 10  # and served


def test_permissive_left_blocked_by_opposing_queue():
    """With the opposing through queue standing, a permissive left may only
    sneak (<= 2 per phase); with no opposing traffic it flows freely."""

    # N carries 260 lefts/h — far above the ~120/h sneakers alone can clear.
    blocked = SimConfig(
        demand=DemandConfig(
            vehicle_rates=(650, 900, 100, 100),  # heavy S opposing N's lefts
            ped_rates=(0, 0),
            turn_fractions=((0.4, 0.5, 0.1), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        ),
        layout=IntersectionLayout(left_turn=(_P, _S, _S, _S)),
    )
    free = SimConfig(
        demand=DemandConfig(
            vehicle_rates=(650, 30, 100, 100),  # almost no opposing traffic
            ped_rates=(0, 0),
            turn_fractions=blocked.demand.turn_fractions,
        ),
        layout=blocked.layout,
    )
    n_left_free = _served_left_fraction(free)
    n_left_blocked = _served_left_fraction(blocked)
    assert n_left_free > 0.9
    assert n_left_blocked < n_left_free - 0.15


def _served_left_fraction(config: SimConfig) -> float:
    from traffic_rl.controllers.fixed_time import NaiveController

    sim = _run(config, NaiveController(), n_steps=3600)
    log = sim.event_log.finalize()
    lefts = log["veh_group"] == left_group(0)
    if lefts.sum() == 0:
        return 1.0
    return float((~np.isnan(log["veh_depart"][lefts])).mean())


def test_shared_left_friction_slows_group():
    """A shared lane with heavy lefts against heavy opposing flow discharges
    slower than a pure through lane under identical totals."""

    base = dict(vehicle_rates=(500, 500, 200, 200), ped_rates=(0, 0))
    no_lefts = SimConfig(demand=DemandConfig(**base))
    with_lefts = SimConfig(
        demand=DemandConfig(
            **base,
            turn_fractions=((0.35, 0.55, 0.10), (0.35, 0.55, 0.10), (0, 1, 0), (0, 1, 0)),
        )
    )
    q_no = _mean_ns_queue(no_lefts)
    q_with = _mean_ns_queue(with_lefts)
    assert q_with > q_no * 1.15


def _mean_ns_queue(config: SimConfig) -> float:
    from traffic_rl.controllers.fixed_time import NaiveController

    sim = _run(config, NaiveController(), n_steps=3600)
    log = sim.event_log.finalize()
    return float(log["step_queues"][:, [0, 1]].sum(axis=1).mean())


def test_ite_formulas_textbook_values():
    # 30 mph: Y = 1 + 13.41 / 6.1 = 3.2 s; 45 mph: 1 + 20.12/6.1 = 4.3 s.
    assert ite_yellow(30.0) == pytest.approx(3.2, abs=0.05)
    assert ite_yellow(45.0) == pytest.approx(4.3, abs=0.05)
    # All-red: (W + 6) / v ; 30 mph, 15 m wide -> 21/13.41 = 1.57 -> 1.6.
    assert ite_all_red(30.0, 15.0) == pytest.approx(1.6, abs=0.05)
    # MUTCD ped clearance: 15 m at 1.2 m/s = 12.5 s.
    assert mutcd_ped_clearance(15.0) == pytest.approx(12.5, abs=0.05)


def test_geometry_flows_into_phases_and_timing():
    geo = GeometryConfig(
        ns_speed_mph=45.0, ew_speed_mph=25.0,
        ns_street_width_m=20.0, ew_street_width_m=12.0,
    )
    layout = IntersectionLayout(left_turn=(_X, _X, _X, _X))
    phases = build_phases(layout, geo)
    by_slot = {p.slot: p for p in phases}
    # NS phases (slots 0, 1) carry the 45 mph yellow; EW phases the 25 mph one.
    assert by_slot[1].yellow == pytest.approx(ite_yellow(45.0))
    assert by_slot[3].yellow == pytest.approx(ite_yellow(25.0))
    assert by_slot[1].yellow > by_slot[3].yellow
    timing = timing_from_geometry(geo)
    # Ped movement 0 crosses the EW street (12 m -> 10 s).
    assert timing.ped_clearance_for(0) == pytest.approx(10.0, abs=0.05)
    assert timing.ped_clearance_for(1) == pytest.approx(16.7, abs=0.05)


def test_starvation_guarantee_with_skipping_controller():
    """A controller that only ever cycles between the two through phases must
    not starve the left phases: the machine's call-wait guarantee force-serves
    any phase whose call has waited max_call_wait. (The green-duration backstop
    alone cannot catch this once there are >2 phases.)"""
    from traffic_rl.controllers.base import Controller

    class ThroughOnlyController(Controller):
        name = "through-only"

        def reset(self, config, rng):
            self.through = [
                i for i, p in enumerate(config.phases) if p.ped_movement is not None
            ]

        def act(self, obs):
            # Switch between the through phases every 20 s, never a left.
            return self.through[int(obs.t / 20.0) % len(self.through)]

    config = _left_demand((_X, _X, _X, _X))
    sim = _run(config, ThroughOnlyController(), n_steps=3600)
    log = sim.event_log.finalize()
    # The invariant is bounded SERVICE INTERVALS: with left-bay calls always
    # pending (demand here keeps the bays occupied), each left phase must get
    # a green at least every max_call_wait plus one transition. (Bounded
    # per-vehicle waits are impossible against an adversarial controller that
    # yanks the green after min green — the machine guarantees service
    # frequency, not throughput.)
    limit = config.timing.max_call_wait + 40.0
    phases_log, t_log = log["step_phase"], log["step_t"]
    left_phases = [i for i, p in enumerate(config.phases) if p.ped_movement is None]
    assert left_phases, "expected protected-left phases in this layout"
    for lp in left_phases:
        times = t_log[phases_log == lp]
        assert len(times) > 5, f"left phase {lp} essentially never served"
        gaps = np.diff(times)
        assert gaps.max() <= limit, (
            f"left phase {lp} went {gaps.max():.0f} s between services"
        )


def test_left_phase_clearances_respected():
    """Yellow/all-red intervals hold per phase on a 4-phase site with
    geometry-derived (non-uniform) timings, under an adversarial controller."""
    from conftest import AdversarialController

    geo = GeometryConfig(ns_speed_mph=45.0, ew_speed_mph=30.0,
                         ns_street_width_m=20.0, ew_street_width_m=15.0)
    config = SimConfig(
        demand=DemandConfig(
            vehicle_rates=(500, 500, 400, 400),
            ped_rates=(30, 30),
            turn_fractions=((0.2, 0.7, 0.1),) * 4,
        ),
        layout=IntersectionLayout(left_turn=(_X, _X, _X, _X)),
        geometry=geo,
        timing=timing_from_geometry(geo),
    )
    controller = AdversarialController()
    controller.n_actions = 4
    sim = IntersectionSim(config)
    obs = sim.reset(3)
    rng = np.random.default_rng(9)
    yellow_runs: dict[int, int] = {}
    prev_state, prev_phase, run = None, None, 0
    for _ in range(7200):
        obs = sim.step(int(rng.integers(0, 4))).obs
        state = int(np.argmax(obs.signal_state_onehot))
        phase = obs.phase
        if state == prev_state and phase == prev_phase:
            run += 1
        else:
            if prev_state == int(SignalState.YELLOW):
                yellow_runs.setdefault(prev_phase, run)
                yellow_runs[prev_phase] = min(yellow_runs[prev_phase], run)
            prev_state, prev_phase, run = state, phase, 1
    phases = config.phases
    for phase_idx, min_run in yellow_runs.items():
        expected = config.timing.yellow_for(phases[phase_idx])
        assert min_run * config.dt >= expected - config.dt - 1e-9, (
            f"phase {phase_idx} yellow {min_run} s < ITE {expected} s"
        )
