from traffic_rl.sim.queues import ApproachQueue


def make_queue(n_vehicles: int) -> ApproachQueue:
    q = ApproachQueue(sat_flow=0.5, startup_lost=2.0)
    for i in range(n_vehicles):
        q.add(veh_id=i, t=float(i) * 0.01)
    q.on_green_start()
    return q


def test_startup_lost_time_then_saturation():
    q = make_queue(10)
    departures_by_second = {}
    for step in range(1, 21):
        departed = q.discharge(green_elapsed=float(step), dt=1.0)
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
        order += q.discharge(green_elapsed=float(step), dt=1.0)
    assert order == list(range(6))


def test_credit_does_not_bank_on_empty_queue():
    q = ApproachQueue(sat_flow=0.5, startup_lost=2.0)
    q.on_green_start()
    for step in range(1, 61):
        assert q.discharge(green_elapsed=float(step), dt=1.0) == []
    # A long empty green must not produce an instant burst when vehicles arrive.
    for i in range(5):
        q.add(veh_id=i, t=60.0)
    burst = q.discharge(green_elapsed=61.0, dt=1.0)
    assert len(burst) <= 1
