import numpy as np

from traffic_rl.config import DemandConfig, SimConfig
from traffic_rl.controllers.actuated import ActuatedController
from traffic_rl.controllers.base import Observation
from traffic_rl.controllers.max_pressure import MaxPressureController

TWO_PHASE_CONFIG = SimConfig(
    demand=DemandConfig(vehicle_rates=(400, 400, 400, 400), ped_rates=(60, 60))
)


def _pad8(values) -> np.ndarray:
    out = np.zeros(8)
    out[: len(values)] = values
    return out


def obs(
    t=100.0,
    queues=(0, 0, 0, 0),
    gaps=(0, 0, 0, 0),
    phase=0,
    green=True,
    phase_elapsed=10.0,
    ped_call=(0, 0),
    mask=(True, True),
):
    phase_onehot = np.zeros(2)
    phase_onehot[phase] = 1.0
    state = np.array([1.0, 0, 0]) if green else np.array([0, 1.0, 0])
    return Observation(
        t=t,
        queue_lengths=_pad8(queues),
        oldest_wait=np.zeros(8),
        time_since_arrival=_pad8(gaps),
        arrivals_last_step=np.zeros(8),
        phase_onehot=phase_onehot,
        signal_state_onehot=state,
        phase_elapsed=phase_elapsed,
        ped_call=np.array(ped_call, dtype=float),
        action_mask=np.array(mask),
        phase_slots=np.array([1, 3], dtype=np.int64),
    )


class TestActuated:
    def make(self):
        c = ActuatedController()
        c.reset(TWO_PHASE_CONFIG, np.random.default_rng(0))
        return c

    def test_holds_during_min_green(self):
        assert self.make().act(obs(phase_elapsed=5.0, queues=(0, 0, 9, 9), gaps=(99,) * 4)) == 0

    def test_gap_out_with_conflicting_call(self):
        assert self.make().act(obs(queues=(0, 0, 4, 0), gaps=(3.5, 3.5, 0, 0))) == 1

    def test_extends_while_platoon_arriving(self):
        assert self.make().act(obs(queues=(2, 0, 4, 0), gaps=(0.0, 1.0, 0, 0))) == 0

    def test_standing_queue_occupies_detector(self):
        # No fresh arrivals, but a queue is still discharging: must not gap out.
        assert self.make().act(obs(queues=(5, 0, 4, 0), gaps=(99, 99, 0, 0))) == 0

    def test_rests_in_green_without_demand(self):
        assert self.make().act(obs(queues=(0, 0, 0, 0), gaps=(99,) * 4)) == 0

    def test_max_out(self):
        assert self.make().act(obs(phase_elapsed=45.0, queues=(9, 9, 1, 0), gaps=(0,) * 4)) == 1

    def test_conflicting_ped_call_counts(self):
        assert self.make().act(obs(queues=(0, 0, 0, 0), gaps=(99,) * 4, ped_call=(0, 1))) == 1


class TestActuatedProtectedLefts:
    """Phase skipping: the defining behavior detection buys a quad-left signal."""

    def make(self):
        from traffic_rl.config import IntersectionLayout, LeftTurnTreatment

        config = SimConfig(
            demand=DemandConfig(
                vehicle_rates=(400, 400, 400, 400),
                ped_rates=(0, 0),
                turn_fractions=((0.2, 0.7, 0.1),) * 4,
            ),
            layout=IntersectionLayout(left_turn=(LeftTurnTreatment.PROTECTED,) * 4),
        )
        c = ActuatedController()
        c.reset(config, np.random.default_rng(0))
        return c, config

    def quad_obs(self, queues8, phase, phase_elapsed=30.0):
        phase_onehot = np.zeros(4)
        phase_onehot[phase] = 1.0
        return Observation(
            t=100.0,
            queue_lengths=np.array(queues8, dtype=float),
            oldest_wait=np.zeros(8),
            time_since_arrival=np.full(8, 99.0),
            arrivals_last_step=np.zeros(8),
            phase_onehot=phase_onehot,
            signal_state_onehot=np.array([1.0, 0, 0]),
            phase_elapsed=phase_elapsed,
            ped_call=np.zeros(2),
            action_mask=np.ones(4, dtype=bool),
            phase_slots=np.array([0, 1, 2, 3], dtype=np.int64),
        )

    def test_skips_empty_left_phase(self):
        c, _ = self.make()
        # On NS-through (phase 1); EW lefts (groups 6,7) empty, EW through has
        # demand -> the EW-left phase (2) is skipped straight to EW-through (3).
        o = self.quad_obs((0, 0, 5, 5, 0, 0, 0, 0), phase=1)
        assert c.act(o) == 3

    def test_serves_left_phase_with_call(self):
        c, _ = self.make()
        # EW left bays occupied -> EW-left phase (2) comes next in order.
        o = self.quad_obs((0, 0, 5, 5, 0, 0, 3, 2), phase=1)
        assert c.act(o) == 2

    def test_left_phase_maxes_out_early(self):
        c, _ = self.make()
        # Left phases run a shorter max green (20 s default). NS-through and
        # EW-left have no calls, so max-out goes straight to EW-through (3).
        o = self.quad_obs((0, 0, 5, 5, 9, 9, 0, 0), phase=0, phase_elapsed=20.0)
        assert c.act(o) == 3


class TestMaxPressure:
    def make(self):
        c = MaxPressureController()
        c.reset(TWO_PHASE_CONFIG, np.random.default_rng(0))
        return c

    def test_picks_higher_pressure_phase(self):
        assert self.make().act(obs(queues=(1, 1, 5, 5))) == 1

    def test_holds_on_tie(self):
        assert self.make().act(obs(queues=(3, 3, 3, 3))) == 0

    def test_decision_interval_holds_between_decisions(self):
        c = MaxPressureController(decision_interval=5.0)
        c.reset(TWO_PHASE_CONFIG, np.random.default_rng(0))
        assert c.act(obs(t=100.0, queues=(9, 9, 0, 0), phase=1)) == 0
        # Pressure flips 2 s later, but the 5 s decision interval hasn't elapsed.
        assert c.act(obs(t=102.0, queues=(0, 0, 9, 9), phase=1)) == 0
        assert c.act(obs(t=105.0, queues=(0, 0, 9, 9), phase=1)) == 1

    def test_no_decision_when_masked(self):
        c = self.make()
        assert c.act(obs(queues=(0, 0, 9, 9), mask=(True, False))) == 0
