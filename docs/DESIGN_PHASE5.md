# Phase 5 design: turning movements, protected lefts, real-world timings

Goal: make the model able to express a *real* signalized intersection — turning
movements, left-turn treatments, clearance intervals from geometry — so the
optimizer's output is implementable, not just directionally right.

## Movements (lane groups)

8 vehicle lane groups, fixed indices:

- `0..3` — through+right group per approach (N, S, E, W). Right turns are
  folded into the through group's volume; RTOR is NOT modeled (stated limit).
- `4..7` — left-turn group per approach (N, S, E, W). Exists only when the
  approach has a left-turn bay; empty otherwise.

Opposing approach pairs: N<->S (0<->1), E<->W (2<->3). A left from approach `a`
conflicts with the through movement of `OPPOSING[a]`.

## Left-turn treatments (per approach)

- `SHARED` — no bay; lefts queue in the through group. The group's discharge
  rate is scaled by a friction multiplier: `f = (1 - p_L) + p_L * c_perm / s`
  where `p_L` is the left share, `s` the group sat flow, `c_perm` the
  permissive capacity (below).
- `PERMISSIVE` — bay, permissive only: the left group discharges during its
  street's through phase by gap acceptance.
- `PROTECTED` — bay + protected-only phase (no permissive filtering).

Permissive capacity (classic gap-acceptance formula, HCM-style):
`c_perm = q * exp(-q * t_c) / (1 - exp(-q * t_f))` veh/s, with `q` = opposing
through demand (EMA of opposing arrivals, tau = 120 s), `t_c = 4.5 s`,
`t_f = 2.5 s`; `c_perm = 1/t_f` when `q -> 0`. Filtering requires the opposing
through queue to be EMPTY; while it discharges there are no gaps. **Sneakers**:
when a phase that served permissive lefts ends (yellow start), up to 2 waiting
lefts per group depart — the vehicles that entered the box waiting for a gap.

## Phase table (canonical slots)

Up to 4 phases in fixed canonical order (leading dual lefts, the standard
lead-lead pattern):

| slot | phase | serves (protected) | permissive lefts | peds |
|---|---|---|---|---|
| 0 | NS-left | left groups of N/S approaches marked PROTECTED | — | none |
| 1 | NS-through | 0, 1 | N/S PERMISSIVE-bay lefts | ped 0 (crosses EW street) |
| 2 | EW-left | left groups of E/W marked PROTECTED | — | none |
| 3 | EW-through | 2, 3 | E/W PERMISSIVE-bay lefts | ped 1 (crosses NS street) |

Phases with nothing to serve are omitted; runtime phase indices are compact
(0..n-1) but each phase knows its canonical `slot`, which is what the RL
featurizer keys on (stable meaning across intersections). The legacy 2-phase
model is exactly this table with no protected lefts and zero left demand.

## Real-world timings (ITE / MUTCD)

Computed per phase from `GeometryConfig` (approach speeds + street widths):

- Yellow: `Y = t_r + v / (2a + 2 G g)` with `t_r = 1.0 s`, `a = 3.05 m/s^2`
  (10 ft/s^2), v = approach speed (85th percentile), clamped to [3.0, 6.0] s.
- All-red: `AR = (w + L_veh) / v`, `L_veh = 6.0 m`, `w` = width of the street
  being crossed, clamped to [1.0, 4.0] s.
- Ped clearance (FDW): crossing distance / 1.2 m/s (MUTCD 3.5 ft/s), walk 7 s.
- Min green: 8 s through phases, 5 s left phases.

Without a geometry config, legacy defaults (Y 3.0, AR 2.0, clearance 13 s)
apply — existing scenarios and published numbers stay reproducible.

## Demand

`DemandConfig` gains `turn_fractions: ((l,t,r) x 4)`, default `(0,1,0)` per
approach (legacy = all through). Left-group arrival rate = approach rate x l
when the approach has a bay; SHARED approaches keep lefts in the through group
(the friction multiplier handles them). Arrival streams: 8 vehicle + 2 ped
independent PCG64 children (paired-seed rules unchanged).

## Left-turn protection warrant (optimizer)

Per site: protect an approach's lefts if, in ANY analysis interval,
`left_volume >= 240 veh/h` or `left_volume * opposing_through_volume >= 50000`
(one opposing lane; 100 000 for 2+ lanes) — the standard cross-product
screening heuristic. Warranted -> PROTECTED all day (hardware/phasing is a
site decision, splits vary by time of day); else bay -> PERMISSIVE.

## Explicit non-goals (stated limits)

RTOR, protected+permissive (FYA) lefts, lagging lefts, U-turns, multi-lane
group split failures, spillback out of bays (bay storage is infinite). The
corridor/network model stays through-only in this phase.

## Consequences

- Observation arrays: queues/waits/gaps/arrivals are (8,); phase onehot and
  action mask are (n_phases,) plus a `phase_slots` map; featurizer scatters
  into 4 canonical slots -> feature dim changes -> both RL policies retrain.
- All controllers generalize: Webster via critical-movement analysis per
  phase; actuated gains phase skipping (no call = skip) and per-phase timers;
  max-pressure sums pressure over served movements.
- The optimizer emits an implementable timing sheet: phases, splits, yellow,
  all-red, walk/FDW per crossing — plus the warrant decision and its trigger.
