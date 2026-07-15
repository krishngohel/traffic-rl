import numpy as np
import pytest

from traffic_rl.config import DemandConfig, SimConfig
from traffic_rl.controllers.base import Controller, Observation


class AdversarialController(Controller):
    """Requests a random (frequently illegal) phase every step."""

    name = "adversarial"

    def reset(self, config, rng):
        self._rng = rng

    def act(self, obs: Observation) -> int:
        return int(self._rng.integers(0, 2))


class StubbornController(Controller):
    """Never volunteers to switch — exists to exercise the backstop."""

    name = "stubborn"

    def act(self, obs: Observation) -> int:
        return obs.phase


@pytest.fixture
def busy_config() -> SimConfig:
    return SimConfig(demand=DemandConfig(vehicle_rates=(500, 500, 500, 500), ped_rates=(80, 80)))


@pytest.fixture
def adversarial() -> AdversarialController:
    c = AdversarialController()
    c.reset(None, np.random.default_rng(7))
    return c


def drive(sim, controller, seed: int, n_steps: int):
    obs = sim.reset(seed)
    controller.reset(sim.config, np.random.default_rng(seed))
    for _ in range(n_steps):
        obs = sim.step(controller.act(obs)).obs
    return sim.event_log.finalize()
