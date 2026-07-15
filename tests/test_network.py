import numpy as np

from traffic_rl.eval.network_harness import (
    NETWORK_SCENARIOS,
    network_controller_registry,
    run_network_controller,
)
from traffic_rl.sim.network import (
    EASTBOUND,
    NetworkConfig,
    NetworkDemandConfig,
    NetworkSim,
)

EAST_ONLY = NetworkDemandConfig(arterial_east=400, arterial_west=0, cross=0, peds=0)
CORRIDOR = NETWORK_SCENARIOS["corridor"]


def drive_network(config, controller_name, seed, n_steps=None):
    registry = network_controller_registry()
    sim = NetworkSim(config)
    controller = registry[controller_name]()
    obs = sim.reset(seed)
    controller.reset(config, np.random.default_rng(seed))
    for _ in range(n_steps or config.n_steps):
        obs = sim.step(controller.act(obs)).observations
    return sim


def test_vehicle_conservation_east_only():
    """Every eastbound vehicle either exits at the far end or is still inside."""
    config = NetworkConfig(demand=EAST_ONLY, n_nodes=3, warmup=0.0, measured=1800.0)
    sim = drive_network(config, "naive", seed=3)
    entered = len(sim.entry_t)
    exited = int((~np.isnan(np.asarray(sim.exit_t))).sum())
    inside = sum(int(q) for node in sim.nodes for q in [len(a) for a in node.queues])
    in_transit = sum(len(d) for node in sim._transit for d in node)
    assert entered == exited + inside + in_transit
    assert entered > 100  # 400 veh/h * 0.5 h


def test_journey_wait_accumulates_across_nodes():
    """With signals held green for the arterial, waits are ~0; the naive plan
    forces multiple red stops, so journey waits must exceed one node's red."""
    config = NetworkConfig(demand=EAST_ONLY, n_nodes=3, warmup=0.0, measured=2400.0)
    sim = drive_network(config, "naive", seed=5)
    done = ~np.isnan(np.asarray(sim.exit_t))
    waits = np.asarray(sim.journey_wait)[done]
    assert len(waits) > 50
    # A single node's worst red is ~35 s; multi-node journeys can exceed it.
    assert waits.max() > 40.0
    assert waits.min() >= 0.0


def test_link_travel_delay_respected():
    config = NetworkConfig(demand=EAST_ONLY, n_nodes=2, link_travel=25.0, warmup=0.0,
                           measured=600.0)
    sim = NetworkSim(config)
    obs = sim.reset(1)
    registry = network_controller_registry()
    controller = registry["naive"]()
    controller.reset(config, np.random.default_rng(1))
    first_downstream_arrival = None
    for _ in range(600):
        result = sim.step(controller.act(obs))
        obs = result.observations
        if first_downstream_arrival is None and len(sim._mirror[1][EASTBOUND]):
            first_downstream_arrival = result.info["t"]
    assert first_downstream_arrival is not None
    assert first_downstream_arrival >= 25.0  # cannot beat the link travel time


def test_network_metrics_and_paired_seeds():
    config = NetworkConfig(demand=CORRIDOR, n_nodes=3, warmup=600.0, measured=1200.0)
    a = run_network_controller(network_controller_registry()["naive"](), config, seed=9)
    b = run_network_controller(network_controller_registry()["naive"](), config, seed=9)
    assert a["p95_wait"] == b["p95_wait"]  # deterministic per seed
    assert np.array_equal(a["veh_waits"], b["veh_waits"])
    assert a["n_vehicles"] > 0 and a["n_peds"] > 0
    assert a["p95_wait"] >= a["mean_wait"] >= 0


def test_greenwave_beats_naive_on_rush():
    """Coordination must pay on the directional-rush corridor."""
    config = NetworkConfig(
        demand=NETWORK_SCENARIOS["corridor_rush"], n_nodes=4, warmup=1200.0, measured=1800.0
    )
    registry = network_controller_registry()
    seeds = [11, 12, 13]
    naive = np.mean(
        [run_network_controller(registry["naive"](), config, s)["p95_wait"] for s in seeds]
    )
    wave = np.mean(
        [run_network_controller(registry["greenwave"](), config, s)["p95_wait"] for s in seeds]
    )
    assert wave < naive


def test_max_pressure_uses_downstream():
    from traffic_rl.controllers.network import NetworkMaxPressureController

    config = NetworkConfig(demand=CORRIDOR, n_nodes=3)
    sim = NetworkSim(config)
    observations = sim.reset(2)
    controller = NetworkMaxPressureController()
    controller.reset(config, np.random.default_rng(2))
    # Force a state: node 1 has EW queue 10, but node 2's eastbound approach is
    # jammed with 50 — downstream pressure should suppress switching to EW.
    import dataclasses

    obs = list(observations)
    obs[1] = dataclasses.replace(
        obs[1],
        queue_lengths=np.array([6.0, 6.0, 0.0, 10.0]),
        action_mask=np.array([True, True]),
        t=100.0,
    )
    obs[2] = dataclasses.replace(obs[2], queue_lengths=np.array([0.0, 0.0, 0.0, 50.0]))
    actions = controller.act(obs)
    # NS pressure at node 1 = 12; EW pressure = (0-0) + (10-50) = -40 => hold NS.
    assert actions[1] == 0


def test_safety_invariants_hold_per_node():
    """Every node still runs the same safety state machine under network load."""
    config = NetworkConfig(demand=CORRIDOR, n_nodes=3, warmup=0.0, measured=2400.0)
    sim = drive_network(config, "max_pressure", seed=21)
    for node in sim.nodes:
        log = node.event_log.finalize()
        states, phases = log["step_state"], log["step_phase"]
        run_start = 0
        for i in range(1, len(states) + 1):
            boundary = (
                i == len(states)
                or states[i] != states[run_start]
                or phases[i] != phases[run_start]
            )
            if boundary:
                length = i - run_start
                final = i == len(states)
                if states[run_start] == 1 and not final:  # yellow
                    assert length >= 3
                if states[run_start] == 2 and not final:  # all-red
                    assert length >= 2
                run_start = i
