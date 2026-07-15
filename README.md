# traffic-rl

**Can smarter traffic-light timing cut how long we all wait?**

We have all sat at a red light on an empty road while the busy direction backs
up. This project tests whether better signal control actually fixes that —
starting with an honest simulator and the four classic strategies traffic
engineers already use, so anything built later has real opponents to beat.

**Phase 2 verdict, up front: at a single intersection the RL agent did not beat
the classics — and that result is published here, not buried.** Details below;
the one place it genuinely wins is near saturation. **Phase 3 moves to a
corridor of four intersections**, where the story shifts: the shared RL policy
beats every *coordination* classic (green wave, network max-pressure) and
statistically ties actuated on the rush corridor.

**This is also a practical tool**: point `traffic-rl-optimize` at a CSV of real
intersection counts and it returns optimized time-of-day signal plans with a
simulated, CI-backed projection of the wait-time reduction. See "Optimize a
real intersection" below.

## Phase 1 results

Headline metric: **p95 wait** — the unluckiest 1 in 20 drivers. Average wait
lies; it hides a light quietly leaving one road to rot. Every number below is
the mean of **20 independent seeded runs** with a t-based 95% CI, and every
controller sees the *same* 20 demand realizations (paired seeds).

![p95 wait by controller and scenario](docs/charts/p95_bar.png)

| p95 wait (s) | symmetric | asymmetric | heavy |
|---|---|---|---|
| Naive 50/50 | 44.7 | **153.2** (CI 127–179, wildly unstable) | 78.0 |
| Webster (1958) | 50.0 | 62.1 | 65.6 |
| Actuated | **37.8** | **35.5** | **50.8** |
| Max-pressure | 42.5 | 36.9 | 68.8 |

On the asymmetric scenario (a busy road crossing a quiet one — the case that
motivated the project), demand-blind 50/50 timing leaves the unlucky driver
waiting **2.5 minutes and swings wildly run to run**, while a 1958 formula cuts
that 2.5x and the two adaptive controllers cut it **4.3x**. Paired per-seed
differences vs naive (the statistically honest comparison): Webster −91 s
[64, 118], actuated −118 s [92, 144], max-pressure −116 s [90, 142] — all
decisive.

Two honest wrinkles the charts surface rather than hide:

- **On symmetric demand, naive is fine.** A 50/50 split is the *right* answer
  when demand is symmetric; Webster lands in the same place. The naive
  controller's failure mode is specifically demand asymmetry.
- **Webster pays pedestrians for its vehicle gains.** The pedestrian service
  floor (20 s walk + clearance) forces Webster onto a long cycle on asymmetric
  demand, and its minor-road pedestrians wait for it (ped p95 85 s vs ~45–51 s
  for everyone else — see `docs/charts/ped_wait_bar.png`).

![Queue growth](docs/charts/queue_timeseries.png)

## Why trust these numbers

- **Per-run p95, aggregated across runs.** Waits within a run are autocorrelated,
  so a pooled percentile has no valid confidence interval. Each run contributes
  one p95; the 20 runs get a t-CI. The pooled distribution is only used for the
  (clearly labeled) ECDF chart.
- **Paired seeds (common random numbers).** Run *k* uses the same seed for every
  controller, and superiority claims cite the CI on the paired per-seed
  differences, not overlapping marginal bars.
- **Censoring is surfaced, not hidden.** Vehicles still queued at the horizon
  count with a lower-bound wait; if more than 5% of a run is censored its p95 is
  reported as "≥" and the run is flagged unstable. Dropping them would flatter
  exactly the worst controllers.
- **Identical rules for everyone.** One warm-up (1200 s), one measured hour, one
  signal-safety state machine (min green, yellow, all-red, ped locks, 120 s
  anti-starvation backstop) shared by all controllers.
- **Safety is enforced by the simulator, not trusted to controllers.** An
  adversarial controller that requests a random phase every second is part of
  the test suite; the state machine makes it safe by construction. The same
  guarantee will hold for the RL agent.
- **Fairness of baseline parameters.** Actuated uses textbook defaults (min
  green 8 s, max 45 s, 3.0 s gap). Max-pressure uses a 15 s control period,
  chosen from a documented sweep ({5, 10, 15, 20} s: heavy-scenario p95 falls
  83→62 s as the period grows while asymmetric stays ~36 s — max-pressure is
  blind to switching cost, and 5 s decisions thrash near saturation).

## The model (and its limits)

Single 4-way intersection, two phases (NS / EW), one lane group per approach,
no protected left turns. **Point-queue model**: Poisson arrivals per approach
(independent RNG streams), saturation-flow discharge (1800 veh/h/lane) with 2 s
startup lost time, no discharge during yellow/all-red. Pedestrians arrive
Poisson per crossing, place a call, and are served concurrently with the
parallel vehicle phase (7 s walk + 13 s clearance, never truncated).

Deliberate simplifications, stated up front: no car-following dynamics, no
spillback or link lengths (irrelevant with one intersection), no left turns,
1 s timestep. These change absolute waits but not controller *rankings*, which
is what Phase 1 is for. Max-pressure at a single isolated intersection honestly
degenerates to weighted longest-queue-first — it is included because it is the
standard classical baseline in the RL traffic literature.

The simulator runs **~40,000x real time** (a full sim-hour in well under a
second of wall clock; the entire 12-experiment, 240-run evaluation takes about
a minute).

## Optimize a real intersection from count data

Feed the tool a standard intersection count study — per-approach volumes per
interval, the data cities already collect:

```csv
time,north_veh,south_veh,east_veh,west_veh,ped_ns,ped_ew
07:00,560,540,180,170,35,30
08:00,640,610,210,190,45,40
09:00,420,400,180,170,30,25
```

```powershell
traffic-rl-optimize counts.csv --runs 10
# benchmark against your intersection's current plan:
traffic-rl-optimize counts.csv --current-greens 25 25
```

It computes a Webster plan per interval, refines each with a paired-seed local
search over cycle and split, assembles time-of-day plans, and evaluates them
against baselines on the full demand profile. Output: `report.md` /
`report.json` (the recommended plans plus a CI-backed comparison) and
`comparison.png`. On the bundled example (`examples/counts_example.csv`, a
5-hour AM-peak profile) the recommended plans cut p95 wait **58%** vs a naive
50/50 signal (42.7 s vs 102.8 s); the report also shows what detection
hardware would buy (actuated: 34.3 s). Projections inherit the model's stated
limits — point-queue, no turning movements — so treat them as a screening
study, not a signed-off timing sheet.

**Whole corridors too:** add a `node` column (0 = west-most signal, increasing
eastward) with counts for every intersection, and the same command retimes the
arterial — coordinated time-of-day plans with a common cycle, per-node Webster
splits, and progression offsets chosen by simulation (eastbound wave, westbound
wave, or simultaneous, per interval):

```powershell
traffic-rl-optimize corridor_counts.csv --link-travel 20
```

On the bundled 3-intersection example (`examples/corridor_counts_example.csv`,
a westbound AM rush) it correctly picks a **westbound** progression for the
peak interval and cuts journey p95 (the sum of a vehicle's waits along the
whole corridor) from an unstable **≥ 790 s** under uncoordinated naive timing
to **74 s** — with the adaptive references (actuated 61 s, shared RL 62 s)
reported alongside.

## Phase 3: four intersections — coordination changes the story

A corridor of 4 signals on an EW arterial (link travel 20 s): eastbound and
westbound traffic traverses every signal, cross streets enter locally, and a
vehicle's wait is the **sum of its queue waits along the whole journey**.
Through-traffic only, no spillback between nodes — stated limits.

| journey p95 (s), 20 paired seeds | corridor | corridor_rush | corridor_cross |
|---|---|---|---|
| Naive 50/50 (uncoordinated) | ≥ 283 | ≥ 1221 (all runs unstable) | 165 |
| Green wave (coordinated Webster + offsets) | 81 | 116 | 101 |
| Max-pressure (downstream-aware) | 125 | 255 | 121 |
| Actuated (independent) | **70** | 86 | **73** |
| **RL (shared policy)** | 75 | **84** | 79 |

![Network p95](docs/charts/network/p95_bar.png)

Three honest findings:

1. **Uncoordinated fixed time is catastrophic at network scale** — queues
   compound across signals until the corridor is fully unstable at rush.
2. **The shared RL policy beats every coordination method.** One set of
   weights runs all four intersections, seeing only local state plus the two
   downstream arterial queues. It beats the green wave and network
   max-pressure on all three corridors, and vs actuated the gap narrows from
   ~30% at one intersection to ~7%: −4.6 s [−6.2, −3.0] on `corridor`,
   −6.1 s [−7.8, −4.3] on `corridor_cross` (where it serves pedestrians
   *better*), and a statistical tie on `corridor_rush` (+1.5 s [−1.2, +4.1]).
3. **Actuated still holds the crown.** Local adaptation with three tuned
   parameters remains unbeaten overall. Max-pressure's poor showing comes with
   a caveat: its optimality theory is about spillback-constrained networks,
   which this model deliberately omits.

Train the corridor policy yourself: `python -m traffic_rl.rl.train_network`.
Evaluate: `traffic-rl-eval-network`.

## Quickstart

```powershell
pip install -e .[dev]          # numpy core + charts + viewer + tests
pytest                         # 35 tests incl. safety invariants + 800x perf gate
traffic-rl-eval --runs 20      # full 4-controller x 3-scenario evaluation
traffic-rl-charts              # renders results/charts/*.png
traffic-rl-watch --controller actuated --scenario asymmetric --speed 8
```

The viewer (`pygame-ce`) shows live queues, signal heads, and pedestrian walk
phases at 1x–1024x. Keys: `Space` pause · `+`/`-` speed · `R` new seed · `Esc`
quit.

## Phase 2: the RL agent — an honest negative result

A double-DQN (pure NumPy: 22→64→64→2, replay, target network, action masking —
no GPU, no framework, fully seeded) was trained for 1.5M steps on **randomized
demand** (per-approach 100–650 veh/h, peds 20–90/h): one policy, no
per-scenario tuning, reward = −(vehicle + pedestrian) waiting per second, so it
cannot win by starving crosswalks. Evaluated by the identical harness, seeds,
and metrics as the classics.

| p95 wait (s), paired vs actuated | symmetric | asymmetric | heavy |
|---|---|---|---|
| Actuated (best classic) | **37.8** | **35.5** | 50.8 |
| RL (DQN) | 48.2 | 46.8 | **49.3** |
| Paired Δ (act − RL), 95% CI | −10.4 [−11.3, −9.4] | −11.3 [−12.4, −10.3] | **+1.5 [+0.3, +2.7]** |

**The verdict:** the RL agent beats naive everywhere and Webster on the hard
scenarios, but a 1960s-era vehicle-actuated controller with three parameters
beats it decisively on two of three scenarios, and it costs pedestrians more
(ped p95 63–74 s vs actuated's 47–49 s) except on heavy. Its one statistically
significant win is the near-saturation scenario (+1.5 s p95 over actuated,
with better ped waits there too) — plausibly because saturation is where
myopic gap-out logic wastes capacity and value estimation helps.

Fairness both ways: the classics got a parameter sweep, so the agent got a
serious retry (3M steps, 128-wide net, γ=0.995, slower exploration decay) — it
came out *worse* on asymmetric (68.7 s). The shipped weights are the better
first run. Things not yet tried that might flip this: longer training with
prioritized replay, a recurrent policy, reward shaping on p95 rather than mean
wait, or multi-intersection settings (where max-pressure's theory shines and
hand-tuned controllers coordinate poorly — likely the more interesting fight).

Train your own: `traffic-rl-train` (~3 minutes on a laptop, deterministic per
seed). The optional Gymnasium wrapper (`pip install traffic-rl[rl]`,
`traffic_rl.rl.env.TrafficEnv`) exists so you can point stable-baselines3 or
any Gym-compatible library at the same sim.

### The RL interface

The sim core *is* the environment: `reset(seed) -> Observation`,
`step(action) -> StepResult`. `Observation` is fixed-size numeric arrays (flattens
straight into a Gymnasium `Box`), `StepResult.info` carries per-step reward
ingredients (`wait_accrued_this_step`, `ped_wait_accrued_this_step`,
`departures_this_step`, `total_queue`), and `action_mask` exposes which phases
are legal. The Gymnasium wrapper needed zero changes to the core — and the
signal state machine means a half-trained policy still cannot run a yellow,
truncate a walk phase, or starve an approach past the backstop.

## Layout

```
src/traffic_rl/
├── sim/          # IntersectionSim, SignalStateMachine, queues, arrivals, NetworkSim
├── controllers/  # naive, Webster, actuated, max-pressure, rl; network: green wave, ...
├── rl/           # NumPy double-DQN, trainers, trained weights, Gymnasium wrapper
├── eval/         # metrics (the honesty rules), harnesses, charts
├── viewer/       # live 3D pygame viewer
├── data.py       # real traffic-count CSV -> demand schedule
└── optimize.py   # traffic-rl-optimize: signal retiming from real counts
```

MIT license. Built as the Phase 1 floor for an open RL-for-traffic experiment.
