from traffic_rl.config import permissive_capacity
from traffic_rl.sim.queues import MovementQueue


def make_queue(n_vehicles: int) -> MovementQueue:
    q = MovementQueue(sat_flow=0.5, startup_lost=2.0)
    for i in range(n_vehicles):
        q.add(veh_id=i, t=float(i) * 0.01)
    q.on_green_start()
    return q


def _sat_discharge(q: MovementQueue, green_elapsed: float, dt: float = 1.0) -> list[int]:
    return q.discharge(q.saturation_service(green_elapsed, dt))


def test_startup_lost_time_then_saturation():
    q = make_queue(10)
    departures_by_second = {}
    for step in range(1, 21):
        departed = _sat_discharge(q, green_elapsed=float(step))
        if departed:
            departures_by_second[step] = departed
    # No flow during the first 2 s; then 0.5 veh/s => one vehicle every 2 s from t=4.
    assert min(departures_by_second) == 4
    assert sorted(departures_by_second) == [4, 6, 8, 10, 12, 14, 16, 18, 20]
    assert all(len(v) == 1 for v in departures_by_second.values())


def test_fifo_order():
    q = make_queue(6)
    order = []
    for step in range(1, 30):
        order += _sat_discharge(q, green_elapsed=float(step))
    assert order == list(range(6))


def test_credit_does_not_bank_on_empty_queue():
    q = MovementQueue(sat_flow=0.5, startup_lost=2.0)
    q.on_green_start()
    for step in range(1, 61):
        assert _sat_discharge(q, green_elapsed=float(step)) == []
    # A long empty green must not produce an instant burst when vehicles arrive.
    for i in range(5):
        q.add(veh_id=i, t=60.0)
    burst = _sat_discharge(q, green_elapsed=61.0)
    assert len(burst) <= 1


def test_friction_multiplier_slows_discharge():
    q = make_queue(10)
    served = []
    for step in range(1, 21):
        served += q.discharge(q.saturation_service(float(step), 1.0, multiplier=0.5))
    # Half rate: ~0.25 veh/s over 18 effective seconds -> 4-5 vehicles, not 9.
    assert 3 <= len(served) <= 5


def test_sneaker_pop_respects_queue_and_order():
    q = make_queue(3)
    assert q.pop(2) == [0, 1]
    assert q.pop(2) == [2]
    assert q.pop(2) == []


def test_permissive_capacity_shape():
    # No opposing traffic: pure follow-up headway (1/2.5 = 0.4 veh/s).
    assert abs(permissive_capacity(0.0) - 0.4) < 1e-9
    # Monotone decreasing in opposing flow, and strictly positive.
    caps = [permissive_capacity(q) for q in (0.05, 0.1, 0.2, 0.4)]
    assert all(a > b for a, b in zip(caps, caps[1:], strict=False))
    assert caps[-1] > 0.0
