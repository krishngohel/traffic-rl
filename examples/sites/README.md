# Sample intersections ("sample lights")

Two ready-to-run intersections with full turning-movement counts and geometry.
**These are representative sites** — volume profiles and dimensions follow
typical published values for their type; they are not surveys of specific
real intersections. Swap in your own count study and geometry to study a real
one.

## suburban_arterial

A 45 mph 4-lane NS arterial crossing a 30 mph collector, left-turn bays on
every approach. Heavy directional lefts (AM southbound, PM northbound) — the
classic case where the protected-left warrant triggers:

```powershell
traffic-rl-optimize examples\sites\suburban_arterial\counts.csv --site examples\sites\suburban_arterial\site.json
```

Expected: NS lefts warrant protection (cross-product), EW bays stay
permissive, a 3-4 phase plan with 4.3 s yellows on the 45 mph street.

## downtown_corner

A 25 mph downtown grid corner: one lane per direction, no bays (lefts share
the lane), pedestrian volumes that peak at lunch. Shared-lane left friction
and the walk/clearance floor dominate the timing here:

```powershell
traffic-rl-optimize examples\sites\downtown_corner\counts.csv --site examples\sites\downtown_corner\site.json
```

Expected: no protected lefts (no bays), 2-phase plan, short 3.2 s yellows,
cycles held up by the pedestrian service floor rather than vehicle demand.

## Legacy examples

`examples/counts_example.csv` (through-only totals) and
`examples/corridor_counts_example.csv` (3-signal corridor) still work
unchanged; through-only data runs on the legacy 2-phase model.
