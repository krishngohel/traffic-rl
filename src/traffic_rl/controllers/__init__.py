from collections.abc import Callable

from traffic_rl.controllers.actuated import ActuatedController
from traffic_rl.controllers.base import Controller, Observation, StepResult
from traffic_rl.controllers.fixed_time import (
    NAIVE_PLAN,
    FixedTimeController,
    FixedTimePlan,
    NaiveController,
)
from traffic_rl.controllers.max_pressure import MaxPressureController
from traffic_rl.controllers.webster import WebsterController, webster_plan

CONTROLLER_REGISTRY: dict[str, Callable[[], Controller]] = {
    "naive": NaiveController,
    "webster": WebsterController,
    "actuated": ActuatedController,
    "max_pressure": MaxPressureController,
}

__all__ = [
    "CONTROLLER_REGISTRY",
    "NAIVE_PLAN",
    "ActuatedController",
    "Controller",
    "FixedTimeController",
    "FixedTimePlan",
    "MaxPressureController",
    "NaiveController",
    "Observation",
    "StepResult",
    "WebsterController",
    "webster_plan",
]
