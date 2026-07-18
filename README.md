# traffic-rl

**Can smarter traffic-light timing cut how long we all wait?**

We have all sat at a red light on an empty road while the busy direction backs
up. This project answers, with evidence, whether better signal control
actually fixes that.

## What is this, in plain terms?

Every traffic light runs on a **timing plan** — a schedule that decides how
many seconds of green each direction gets before the light switches. Most
lights in the world run surprisingly simple plans, often set years ago and
never revisited. Bad timing is invisible money: thousands of people idling a
few extra seconds, every cycle, all day.

This project is three things in one repo:

1. **A traffic simulator.** A virtual four-way intersection (and a corridor of
   several intersections in a row) where cars arrive randomly at realistic
   rates, queue up, and drive off when they get green — plus pedestrians who
   push the button and cross. It runs about 40,000× faster than real time, so
   you can test a year of traffic in minutes.
2. **A competition between ways of controlling the light.** The classic
   strategies traffic engineers actually use — from "give everyone equal
   green" to detector-driven controllers — compete against an AI that taught
   itself by trial and error. Everyone plays by the same rules on the same
   traffic, and the scoreboard is published even when the AI loses.
3. **A practical retiming tool.** Point `traffic-rl-optimize` at a
   spreadsheet of traffic counts (the data cities already collect) and it
   returns a study: how long people wait under your current timings, a better
   timing plan if one exists, and how much waiting it would save — with error
   bars.

## The results, in plain English

A quick guide to reading the numbers:

- **"p95 wait"** is the wait of the unluckiest 1-in-20 drivers. Averages hide
  pain: a light can have a fine *average* while some drivers sit through
  multiple red cycles. p95 is how long the unlucky ones wait.
- **Every strategy faces the exact same traffic.** Each test is run 20 times
  with different random traffic, and run #7 uses the *same* cars arriving at
  the *same* moments for every strategy — so differences come from the
  strategy, not luck.
- **Numbers in brackets like [4.6, 15.5]** are 95% confidence intervals:
  statistics' way of saying "the true improvement is very likely between
  these two values." If a claimed win's interval included zero, we would not
  call it a win.

| What was tested | Waiting before | Waiting after | Change |
|---|---|---|---|
| A 3-light corridor over a morning rush: coordinated retiming vs. leaving each light on a basic equal-split plan | 764 s (p95, whole trip) | **69 s** | **−91%** |
| One intersection over a 5-hour morning: this tool's recommended plans vs. a basic 50/50 light | 101.5 s (p95) | **42.3 s** | **−58%** |
| The same study measured against *realistic* current timings (what a competent signal shop would already have installed) | 44.2 s (p95) | **42.3 s** | −4% — honest: standard practice already captures most of the easy gains |
| The AI improving itself: two minutes of self-practice on its worst intersection, then scored on traffic it had never seen | 54.8 s (average) | **29.3 s** | **−47%** [23.2, 27.9] |
| The AI vs. the best classic controller on the rush-hour corridor (the AI's one clear win) | 88.4 s (p95, whole trip) | **78.4 s** | **−10.0 s** [4.6, 15.5] |
| Time to run the full evaluation suite (32-core desktop, identical results) | ~10 min | **< 1 min** | 7–15× |

And the honest counterweights, published with the same care: at a single
isolated intersection, the classic detector-driven ("actuated") controller —
1960s technology — **still beats the AI in every scenario** (34.1 s vs 36.5 s
p95 on the morning example). And when the optimizer cannot beat an
intersection's current plan, it says **"no retiming warranted"** instead of
manufacturing an improvement.

## How it works

### The simulated intersection

A four-way crossing with up to eight **lane groups**: for each of the four
approaches, a through+right group and a left-turn group. Left turns can be
handled three ways, just like real streets: sharing the through lane, yielding
through gaps in oncoming traffic ("permissive"), or getting their own
exclusive green arrow ("**protected left**"). Cars arrive at random following
realistic hourly rates, join a queue, and discharge at real-world saturation
rates when their light turns green. Pedestrians place calls and get walk
signals with legally-required minimum crossing times.

Safety is built into the simulator itself, not trusted to any controller: no
matter what a controller asks for, the light always serves minimum greens,
yellow and all-red clearance intervals computed from the street's real
geometry (approach speed, crossing width — the same ITE/MUTCD formulas
practitioners use), and a hard guarantee that no direction can be starved for
more than 200 seconds. An AI mid-training can request nonsense; the virtual
light stays safe.

### The contenders

- **Naive equal-split** — every direction gets the same green, always. The
  "never been retimed" baseline.
- **Webster (1958)** — the classic formula: observe traffic for 15 minutes,
  then set green times proportional to demand. What a careful engineer
  computes by hand.
- **Actuated** — the detector-driven controller on most real suburban
  signals: pavement sensors extend the green while cars keep coming and cut
  it short when they stop. 1960s technology, and still the one to beat.
- **Max-pressure** — a modern academic algorithm that switches based on queue
  imbalance.
- **RL (the AI)** — a small neural network (pure NumPy, no GPU needed) trained
  by reinforcement learning: it played millions of simulated seconds of
  traffic, was rewarded for keeping total person-waiting low, and learned its
  own switching strategy. One set of weights handles any intersection layout.

### Watch it live

`traffic-rl-watch` opens a 3D viewer: cars drive in, queue, and visibly turn
left/right/straight through the intersection across three lanes; protected
lefts get their own green-arrow signal heads; each light shows a floating card
with its current green timings; a HUD tracks waits in real time at 1× to
1024× speed.

`traffic-rl-watch --learn` is the same viewer with the AI **learning while
you watch**: it keeps training on the live traffic, and the rolling
mean-wait readout falls as it improves. The −47% self-improvement result in
the table above is this mode, validated afterward on unseen traffic.

### The retiming tool

Feed `traffic-rl-optimize` a CSV of traffic counts — either simple hourly
totals per approach, or a full **turning-movement count** (left/through/right
per approach, the standard format traffic studies produce):

```csv
time,north_left,north_thru,north_right,south_left,...,west_right,ped_ns,ped_ew
07:00,215,610,60,95,340,45,...,25,18,14
```

```powershell
traffic-rl-optimize counts.csv --site site.json
# benchmark against your intersection's actual current plan:
traffic-rl-optimize counts.csv --site site.json --current-greens 15 30 28
```

**The study starts where the street starts.** The baseline is the
intersection's *current* timings: pass `--current-greens` for the installed
plan, or the tool derives what a typical signal shop would have installed —
a cycle length from the Federal Highway Administration's published 60–120 s
practice range, green splits proportional to your peak counts. It measures
how long people wait under that plan **first** (graded A–F on the Highway
Capacity Manual's level-of-service scale), then optimizes. The current plan
also enters the final tournament, so the recommendation can never lose to
what is already installed.

What the study contains:

- **Left-turn protection screening** — should any approach get a green
  arrow? Checked against the standard volume thresholds, with the triggering
  numbers cited.
- **A phase plan with clearance intervals computed from your geometry** —
  yellow time from approach speed, all-red from crossing width, pedestrian
  walk times from crosswalk length (`site.json` carries the geometry;
  without it, conservative defaults are used and the report says so).
- **Time-of-day green splits** — different plans for different hours, refined
  by simulation, then a full-day tournament (because optimizing each hour
  separately misses queues that carry over from the peak — measured, not
  assumed).
- **A fair comparison table** — your current plan, the recommendation, the
  naive light, Webster, actuated, max-pressure, and the AI, all with error
  bars — because "what would detection hardware buy instead?" is the question
  a screening study should answer.

**Whole corridors too:** add a `node` column and the same command retimes an
arterial — one shared cycle length, per-intersection green splits, and
progression offsets (the "green wave" that lets a platoon of cars hit
consecutive greens) chosen by simulation. On the bundled 3-intersection
example it cuts whole-trip p95 wait from an unstable ≥ 764 s under
uncoordinated naive timing to 69 s (actuated reaches 61 s, the shared AI 59 s
alongside — the corridor model stays through-only, a stated limit).

**Sample data**: `examples/` bundles a 5-hour morning count profile, a
corridor version, and two fully-specified intersections in
`examples/sites/` — a 45 mph suburban arterial whose left-turn volumes
trigger the protection warrant, and a 25 mph downtown corner with shared-lane
lefts and lunch-peak pedestrians.

## The detailed scoreboard

Headline metric: **p95 wait** — the unluckiest 1 in 20 drivers. Every number
is the mean of **20 paired-seed runs** with a t-based 95% CI; every controller
sees the same 20 traffic realizations. "≥" marks runs where some vehicles were
still queued when the simulation ended, so the true number is at least this
big; "unstable" means the queue was still growing.

| p95 wait (s) | symmetric | asymmetric | heavy | arterial_lefts | downtown_shared |
|---|---|---|---|---|---|
| Naive equal-split | 45.8 | **≥ 162.4** | 80.5 | 68.0 | 39.9 |
| Webster (1958) | 50.3 | 62.0 | 68.3 | ≥ 137.1 (unstable) | 51.5 |
| **Actuated** | **37.6** | **36.0** | **51.0** | **54.1** | **34.5** |
| Max-pressure | 43.9 | 37.1 | 69.0 | 94.3 | 36.9 |
| RL (the AI) | 40.4 | 37.3 | 53.9 | 202.8 | 37.5 |

The five scenarios: balanced demand, heavily lopsided demand, near-capacity
demand on all approaches, a 45 mph arterial with protected lefts and
geometry-derived timings (`arterial_lefts`, the flagship), and a narrow
downtown corner where lefts share lanes and pedestrians dominate. Paired
differences (actuated − RL): −2.8 [−3.6, −2.1] symmetric, −1.3 [−2.4, −0.2]
asymmetric, −2.9 [−4.1, −1.7] heavy, −3.0 [−3.9, −2.1] downtown — actuated's
wins are small but statistically real. On the arterial it is −149 s: not
close, and the honest headline of the realism upgrade.

![p95 wait by controller and scenario](docs/charts/p95_bar.png)

Two findings worth the price of admission:

1. **Webster goes unstable at the protected-left arterial** — a single fixed
   plan built from average flows cannot give the left-turn arrow enough green
   at the peak without wasting green all day. This is exactly why real
   arterials get detection hardware, and the model reproduces it.
2. **The AI's arterial failure has a specific cause, not a mystery.** Serving
   the left-turn phase costs several seconds of clearance time *now* while
   the payoff (a short left queue) accrues slowly — so the policy learned to
   defer it, badly. During training this starved left turns outright, which
   is how a real safety gap in the signal logic was found and fixed (see
   below).

### Corridor results (through-only model, 20 paired seeds)

Here the metric is **journey p95** — a driver's total waiting summed across
every light on their trip through four intersections.

| journey p95 (s) | corridor | corridor_rush | corridor_cross |
|---|---|---|---|
| Naive 50/50 (uncoordinated) | ≥ 273 | ≥ 1213 (all runs unstable) | 157 |
| Green wave (coordinated Webster + offsets) | 80.3 | 117.7 | 100.7 |
| Max-pressure (downstream-aware) | 127.2 | 256.8 | 123.6 |
| Actuated (independent) | **70.5** | 88.4 | **74.0** |
| **RL (one shared policy)** | 73.6 | **78.4** | 76.9 |

![Network p95](docs/charts/network/p95_bar.png)

One set of AI weights runs all four intersections, each seeing only its own
queues plus two downstream ones. Paired vs actuated: **+10.0 s [+4.6, +15.5]
on corridor_rush** — a statistically decisive win exactly where coordination
between lights matters most — and −3.0 / −2.9 s (small losses) on the other
two. The thesis after realism: **learning wins where coordination is the
problem; tuned local adaptation still wins where phase discipline is.**

## The AI's story, honestly

An earlier version of the AI (on the simpler, no-turns model) beat actuated
outright — the first ML win of the project. **That crown did not survive the
realistic model.** What happened, documented rather than buried:

- The realism upgrade moved the AI's interface to canonical phase slots
  (NS-left / NS-through / EW-left / EW-through), so one set of weights can
  run any intersection layout.
- Retrained policies stayed strong at conventional intersections but
  repeatedly failed the protected-left arterial. Documented dead ends: a
  fairness term in the reward destabilized training; fully randomized
  geometry diluted the small network; unwinnable oversaturated training
  episodes poisoned the replay buffer (a specialist trained *only* on
  arterial traffic scored p95 2400 s — the smoking gun; fixed by rescaling
  every training episode into a winnable demand band).
- Training-time starvation exposed a **real safety gap**: the old
  anti-starvation rule (force a switch after 120 s of continuous green)
  cannot protect a third phase from a controller that ping-pongs between the
  other two. The state machine now force-serves any waiting call within
  200 s, verified with an adversarial controller that tries to starve it.
- The shipped policy (pure-NumPy double-DQN, trained across randomized
  layouts and demand) lands within 1.3–3 s of actuated everywhere except the
  arterial, keeps the corridor win — and the `--learn` mode above shows the
  arterial gap closing with online self-practice (54.8 → 29.3 s mean wait,
  actuated at 24.0 s).

## Why trust these numbers

- **Per-run p95, aggregated across runs.** Waits within one run influence
  each other, so pooling them and taking a percentile gives invalid error
  bars. Each run contributes one p95; the 20 p95s get a proper t-interval.
- **Paired seeds (common random numbers).** Run *k* uses the same traffic for
  every controller, and superiority claims cite the interval on the paired
  per-run differences — not eyeballed overlapping bars.
- **Censoring is surfaced, not hidden.** Vehicles still queued when the
  simulation ends count with a lower-bound wait; if more than 5% of a run is
  censored, its p95 is reported as "≥" and the run is flagged unstable.
- **Identical rules for everyone.** One warm-up period, one measured window,
  one signal-safety state machine shared by every controller.
- **Safety enforced by the simulator, not trusted to controllers.** The test
  suite drives the light with adversarial controllers designed to starve
  phases; the machine keeps them safe by construction.
- **Fair baseline tuning.** Actuated uses textbook settings plus the phase
  skipping real detection hardware enables; max-pressure's control period
  comes from a documented sweep. When the AI lost, it got serious retries
  (documented above) — the same effort the classics received.

## The model and its limits

Point-queue dynamics: cars arrive via independent random streams per lane
group, queue vertically (no physical length), and discharge at 1800 vehicles
per hour per lane with 2 s startup lost time. Deliberate simplifications,
stated up front: no right-turn-on-red, no flashing-yellow-arrow phasing, no
lagging lefts, left-turn bays never physically overflow, no car-following or
spillback between intersections, 1-second timestep. The corridor model is
through-only. These change absolute wait values, not the honesty of the
comparisons — treat optimizer output as a screening study, not a signed-off
timing sheet.

## History: how the project got here

All numbers below are from the earlier, simpler no-turns model (tagged
releases v0.1.0–v0.5.0 carry the full write-ups):

- **Phase 1** (v0.1.0): simulator + four classic baselines under the honesty
  rules. On lopsided demand, a naive 50/50 light leaves the unlucky driver
  waiting 2.5 minutes; actuated cuts it 4.3×.
- **Phase 2** (v0.2.0): first neural-network controller — an honest negative
  result. Beat naive and Webster, lost to actuated.
- **Phase 3** (v0.3.0–v0.4.0): corridor simulation + the real-data optimizer
  tool; shared AI beat every coordination method, actuated stayed champion.
- **Phase 4** (v0.5.0): pattern-aware policy beat actuated with statistical
  significance — the first ML win. The realistic model (Phase 5) has since
  raised the bar; that fight continues above.

## Quickstart

```powershell
pip install -e .[dev]          # numpy core + charts + viewer + tests
pytest                         # 86 tests incl. safety invariants + perf gate
traffic-rl-eval --runs 20      # 5-controller x 5-scenario evaluation
traffic-rl-charts              # renders results/charts/*.png
traffic-rl-optimize examples\sites\suburban_arterial\counts.csv --site examples\sites\suburban_arterial\site.json
traffic-rl-watch --controller actuated --scenario arterial_lefts --speed 8
traffic-rl-watch --learn --scenario arterial_lefts   # watch the AI improve itself
```

Viewer keys: `Space` pause · `+`/`-` speed · `R` new seed · `Esc` quit.
In `--learn` mode, weights autosave to `results/online_weights.npz` (every
100k steps and on exit); evaluate them with
`RLController(weights="results/online_weights.npz")`.

Every seeded run is independent, so `traffic-rl-eval`,
`traffic-rl-eval-network`, and `traffic-rl-optimize` spread runs across all
CPU cores by default — results are bit-identical to a serial run, just
10–20× faster on a desktop. `--jobs 1` forces serial; `--jobs N` caps
workers.

Train your own policies: `traffic-rl-train` (standard DQN, minutes on a
laptop, deterministic per seed), `traffic-rl-train --pattern` (the
pattern-aware recipe), `python -m traffic_rl.rl.train_network` (corridor
policy). The Gymnasium wrapper (`pip install traffic-rl[rl]`,
`traffic_rl.rl.env.TrafficEnv`) exposes the same sim to stable-baselines3 or
any Gym-compatible library.

## Small glossary

- **Cycle** — one full rotation through all the light's phases.
- **Phase** — a set of movements that get green together (e.g. "north-south
  through traffic").
- **Green split** — how the cycle's green time is divided among phases.
- **Protected left** — a left turn served by its own green arrow, with
  oncoming traffic stopped.
- **Actuated control** — green times that stretch or shrink live based on
  pavement detectors.
- **p95 wait** — the wait exceeded by only 1 in 20 drivers; the "unlucky
  driver" metric.
- **95% confidence interval** — the range the true value falls in with 95%
  probability, given the observed runs.
- **Paired seeds** — every strategy is tested on the exact same random
  traffic, so comparisons cancel out luck.
- **Reinforcement learning (RL)** — training by trial and error against a
  reward signal, here: minimize total person-waiting.

## Layout

```
src/traffic_rl/
├── sim/          # IntersectionSim (8 lane groups), SignalStateMachine, NetworkSim
├── controllers/  # naive, Webster, actuated (w/ phase skipping), max-pressure, rl
├── rl/           # NumPy double-DQN, features, trainers, online learner, Gym wrapper
├── eval/         # metrics (the honesty rules), harnesses, parallel runner, charts
├── viewer/       # animated 3D pygame viewer (turning cars, timing cards, --learn)
├── config.py     # layouts, phase tables, ITE/MUTCD timing formulas
├── data.py       # count CSVs (incl. TMC) + site geometry + left-turn warrant
└── optimize.py   # traffic-rl-optimize: retiming studies from real counts
```

MIT license. Built as an open experiment in honest RL-for-traffic; the
realistic model made it a screening tool a practitioner can argue with.
