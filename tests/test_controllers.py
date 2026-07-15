import numpy as np

from traffic_rl.controllers.actuated import ActuatedController
from traffic_rl.controllers.base import Observation
from traffic_rl.controllers.max_pressure import MaxPressureController


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
        queue_lengths=np.array(queues, dtype=float),
        oldest_wait=np.zeros(4),
        time_since_arrival=np.array(gaps, dtype=float),
        arrivals_last_step=np.zeros(4),
        phase_onehot=phase_onehot,
        signal_state_onehot=state,
        phase_elapsed=phase_elapsed,
        ped_call=np.array(ped_call, dtype=float),
        action_mask=np.array(mask),
    )


class TestActuated:
    def make(self):
        c = ActuatedController()
        c.reset(None, np.random.default_rng(0))
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


class TestMaxPressure:
    def make(self):
        c = MaxPressureController()
        c.reset(None, np.random.default_rng(0))
        return c

    def test_picks_higher_pressure_phase(self):
        assert self.make().act(obs(queues=(1, 1, 5, 5))) == 1

    def test_holds_on_tie(self):
        assert self.make().act(obs(queues=(3, 3, 3, 3))) == 0

    def test_decision_interval_holds_between_decisions(self):
        c = MaxPressureController(decision_interval=5.0)
        c.reset(None, np.random.default_rng(0))
        assert c.act(obs(t=100.0, queues=(9, 9, 0, 0), phase=1)) == 0
        # Pressure flips 2 s later, but the 5 s decision interval hasn't elapsed.
        assert c.act(obs(t=102.0, queues=(0, 0, 9, 9), phase=1)) == 0
        assert c.act(obs(t=105.0, queues=(0, 0, 9, 9), phase=1)) == 1

    def test_no_decision_when_masked(self):
        c = self.make()
        assert c.act(obs(queues=(0, 0, 9, 9), mask=(True, False))) == 0
