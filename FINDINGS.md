# Multi-Agent Disaster Relief System — Validation Findings

## Headline Result

Mixed-tier model routing (local model for field-report parsing, Claude Haiku
for depot negotiation) matches full-Haiku quality while costing roughly
**1/3 as much**, and this holds up across three structurally different
disaster types with proper multi-seed, multi-repeat validation — not a
single-scenario artifact.

> **Update:** a real-disaster stress test (see "Real-Disaster Stress Test"
> below) surfaced a genuine bug in `DepotAgent`'s allocation logic (Issue
> #5) that was present for the numbers below too, not just the stress test.
> The qualitative conclusion (mixed ≈ full-Haiku quality at a fraction of
> the cost) held up after the fix, but this table has not yet been re-run
> with the fix applied — see Open Items. Treat these exact percentages as
> pre-fix until that re-run happens.

| config | flood gap | wildfire gap | earthquake gap | pooled gap | pooled unmet need | cost/run | scenario range |
|---|---|---|---|---|---|---|---|
| local_everywhere | +63.2% | +14.8% | +11.5% | **+29.9% ± 24.5** | 1239.3 | $0.000 | 51.7pp |
| haiku_everywhere | +6.9% | +6.9% | -6.6% | **+2.4% ± 9.7** | 1086.7 | $0.054 | 13.5pp |
| local_parsing_haiku_negotiation | -11.4% | +11.0% | -5.6% | **-2.0% ± 11.3** | 1107.7 | $0.018 | 22.4pp |

(gap% = optimality gap vs. a centralized LP optimizer baseline with full
information; negative = beats the optimizer, positive = loses to it.
5 repeats per config per scenario, 3 disaster types.)

**Conclusions supported by this result:**

1. **Local-only negotiation is a genuinely inferior choice, not scenario
   noise.** `local_everywhere` loses to the optimizer in all three disaster
   types (+63%, +15%, +12%) — the same field-report parsing and dispatcher
   logic as the other two configs, only negotiation differs. Since flood and
   wildfire's parsing task is identical across configs, this isolates the
   failure to the negotiation/allocation judgment task specifically, not
   general model weakness.
2. **Mixed routing is statistically indistinguishable from full Haiku in
   quality** (-2.0% vs +2.4%, well within each other's std) **at ~1/3 the
   cost** ($0.018 vs $0.054 per run). This was the project's original
   cost-efficiency hypothesis, now validated across 3 disaster types with
   5 repeats each rather than a single flood-only run.
3. Mixed routing is somewhat more scenario-sensitive than full Haiku
   (22.4pp range vs. 13.5pp) — beats the optimizer on flood, loses on
   wildfire — but the swings are modest, not dramatic, once the
   methodology issues below were fixed.

---

## Methodology

- **Three disaster types**, each with a distinct damage model
  (`simulation/world.py`), to test whether a config's ranking generalizes
  across qualitatively different crises rather than one convenient scenario:
  - **Flood** — damage accrues gradually and randomly across the map.
  - **Wildfire** — damage spreads outward from a fixed epicenter and
    worsens every timestep; a predictable, escalating threat.
  - **Earthquake** — damage is front-loaded (routes blocked, depot stock
    partially destroyed at t=0), then quiet aftershocks; a sudden-onset
    shock rather than a slow build.
- **Two independent noise sources, measured separately:**
  - `--repeats N` — same scenario, N runs, quantifies model-sampling
    variance (hosted models are non-deterministic even at temperature 0).
  - `--scenarios TYPE:SEED ...` — different scenarios, quantifies whether a
    config's ranking is a general property or a one-seed artifact.
- **Sonnet-based configs were dropped from consideration.** Forced
  extended-thinking collided with a fixed token budget, causing recurring
  empty-response truncation failures that inflated effective cost roughly
  2x through wasted retries. Still selectable via `--configs` for anyone
  who wants to revisit that decision, but excluded from the finalist
  comparison above.

---

## Real-Disaster Stress Test (LA Wildfires, January 2025 calibration)

The toy-scale sweep above validated the finding across three disaster
*types*, but at synthetic scale (6x6 grid, 4 shelters, 15 timesteps). To
test whether it holds at realistic scale/speed/scarcity, a wildfire
scenario was calibrated against actual reporting on the January 2025 LA
(Palisades + Eaton) fires: 200,000+ evacuated, 17,000+ homes destroyed,
hurricane-force wind-driven spread, 24-day containment, and documented
resource strain (188,500 meals across ~178,000 people — thin, not
comfortable). See `run_real_disaster.py`'s module docstring for the full
parameter-by-parameter calibration reasoning.

### Calibration attempt #1: saturated the metric (10x population surge)

First pass scaled shelter population 10x with depot stock left unscaled
(intentionally, to represent relief supply not scaling with an evacuation
surge). Result: `local_everywhere`'s gap dropped from the toy-scale +14.8%
to +1.2% — which looked like great news but wasn't. Diagnostic: starting
depot stock covered only **~1.03 timesteps of total demand out of 24**,
i.e. near-total resource exhaustion almost immediately, for every config
including the optimizer. This reproduces the exact "unrecoverable
stockout" pathology from Issue #1 below — once nobody has anything left to
allocate, there's no decision left for a good vs. bad negotiator to differ
on, so gaps collapse toward each other by construction, not because the
finding generalized.

**Fix:** dropped the surge factor to 4x. Verified via the optimizer's own
stock trace that this produces a genuine, *gradually widening* deficit
(stock depletes to zero around timestep 18 of 24, not timestep 1-2) rather
than instant collapse — both more realistic (real relief was strained and
worsening, not instantly zero) and keeps the comparison metric sensitive
for most of the run. `run_real_disaster.py` now prints a pre-flight
stock/demand ratio and warns if it's heading for saturation again, so this
doesn't require a full run to discover next time.

### Calibration attempt #2, first pass: all three configs converged (+3.1–3.5%)

With the 4x surge, all three finalist configs landed within a suspiciously
tight band:

| config | toy-scale wildfire gap | real-disaster gap | cost/run |
|---|---|---|---|
| local_everywhere | +14.8% | +3.5% (n=1) | $0.000 |
| haiku_everywhere | +6.9% | +3.2% ± 0.1 | $0.238 |
| local_parsing_haiku_negotiation | +11.0% | +3.3% ± 0.0 | $0.110 |

Ranking order was preserved (haiku best, mixed close behind, local worst),
and mixed still matched haiku's quality at 46% of the cost — consistent
with the headline finding. But three architecturally very different
configs landing within 0.3 percentage points of each other, with nearly
identical `cumulative_unmet` values at *every single timestep*, was too
tight to accept at face value without checking why.

### Issue #5: DepotAgent's clamp had a positional bias, independent of model quality (found, fixed)

Investigating the convergence found a real bug in
`DepotAgent._extract_and_validate`, unrelated to any model's actual
negotiation quality. The post-hoc clamp (added originally as a defensive
measure against hallucinated over-allocation) walked shelters in fixed
dict-insertion order (always S1, S2, ..., S10) and subtracted each
shelter's requested amount from remaining stock in that order. Confirmed
with a deterministic unit test with no LLM involved: when 10 equally-urgent
shelters submit identical requests that together exceed available stock,
the clamp gave shelter S1 **100%** of its request and every other shelter —
including ones just as urgent — **0%**, purely because of iteration order.
This is a silent override of whatever the negotiator actually decided, and
it triggers whenever multiple shelters compete for scarce stock in the same
timestep — routine under real scarcity, not an edge case.

**Fix:** replaced the sequential clamp with proportional scaling. If a
resource's total requested amount exceeds available stock, every shelter's
request is now scaled down by the same factor rather than fully satisfying
some and zeroing others by position. Re-verified with the same test: 10
identical requests now each receive an equal 1/10th share; a shelter
requesting 9x more than another still receives ~9x more after scaling
(relative prioritization preserved, not flattened). This bug was present
for the entire toy-scale sweep above too, just triggered less often since
that scenario had more slack — see Open Items for the follow-up re-run.

### Issue #6: the convergence persisted after the fix — a real scarcity-ceiling effect, not a bug

Re-ran the same real-disaster comparison with the fix applied. **All three
configs converged to *exactly* +3.2%** — tighter than before, not looser.
This ruled out Issue #5 as the explanation for the convergence (the fix was
still correct and necessary on its own merits, just not the cause of this
particular effect). A dedicated diagnostic (`run_scarcity_diagnostics.py`)
tested the actual mechanism directly:

| config | mean % shelters critical | first timestep ≥90% critical | total delivered (24 timesteps) | final cumulative unmet |
|---|---|---|---|---|
| optimizer_baseline | 99.6% | 0 | 2,755.7 | 297,713.7 |
| local_everywhere | 99.6% | 0 | 2,803.5 | 307,327.4 |
| haiku_everywhere | 99.6% | 0 | 2,868.9 | 307,169.5 |
| local_parsing_haiku_negotiation | 99.6% | 0 | 2,869.4 | 307,238.9 |

Two confirmed findings:

1. **Virtually every shelter is "critical" (urgency ≥ 0.5) from timestep 0**,
   across every config including the optimizer. There's no meaningful
   "who deserves it more" distinction left for any negotiator, good or bad,
   to exploit — almost everyone already qualifies.
2. **All three agent configs deliver essentially the same total quantity of
   resource** over the full run (2,803–2,869, within ~2.3% of each other),
   which is what actually drives the near-identical final unmet-need
   numbers (within ~0.05% of each other) — not the negotiator's decision
   quality. Total deliverable resource is capped by stock + resupply +
   transport capacity, which are identical across configs; there's no
   volume left over for a smarter negotiator to move more of.
   (An initial version of this diagnostic measured "delivered as % of stock
   available that same timestep," which looked inconsistent — 29–60%
   across configs — but that metric double-counts undelivered leftover
   stock as "available" again on later timesteps, so it doesn't cleanly
   isolate the effect. Absolute total delivered, above, is the correct
   comparison.)

**Interesting side-finding:** the optimizer beats every agent config while
delivering *less* total resource (2,755.7 vs. 2,803–2,869). Its ~3.2% edge
comes from surgically better targeting of the same or less supply, not from
moving more volume — a more precise characterization of what "optimality
gap" represents at this scarcity level than assumed going in.

**Conclusion:** the mixed-routing cost-efficiency finding is real at
moderate scarcity (the toy-scale sweep), but becomes unmeasurable once a
disaster is severe enough that total deliverable supply — not negotiation
quality — is the binding constraint. This is a genuine property of the
scenario, confirmed by direct measurement, not an artifact of either the
clamp bug or the earlier saturation miscalibration.

---

## Phase E/B/A: Richer Negotiation Context, Route-Awareness, Real Transit Delay

Following the scarcity-ceiling finding above, the project shifted from
*validating* the existing architecture to *building it out* — closing
known capability gaps rather than just measuring around them. Three phases,
built and tested in order:

**Phase E — richer negotiation context.** `DepotAgent` previously saw only
a 0-1 urgency float per resource, nothing else. Two additions: (1)
`FieldReportAgent` now also extracts a population estimate from the same
noisy situation-report text it already parses (the text always contained a
population figure; it was simply being discarded) — routed through the
same fog-of-war channel as urgency, not handed over as ground truth. (2)
Fixed relative resource-criticality weights (medical=3.0, water=2.0,
food=1.0) given to the negotiator as domain knowledge, since triage
priority isn't something a field report conveys.

This surfaced a real, pre-existing inconsistency: `optimizer/baseline.py`'s
LP objective had **no population weighting at all**, while the evaluation
metric (`unmet_urgent_need`) already was population-weighted — meaning the
"optimal" baseline wasn't actually optimizing for what was being measured.
Fixed both the LP objective and the metric to use population and
criticality weighting consistently. Verified with a controlled test: two
shelters, identical urgency, population 500 vs. 50 — the old objective
split allocation evenly; the fixed one gives it entirely to the 500-person
shelter, which is what "optimal" should mean given what's scored.

**Phase B — route-aware negotiation.** `DepotAgent.allocate()` now
receives `route_status` (reachable/blocked) per shelter, computed the same
way the optimizer already does (`world.route_damaged()`). Enforced
defensively in code, not just via prompt instruction: verified with a test
where the model *tried* to allocate to a blocked shelter anyway — the
blocked shelter still received zero, and the freed stock went to the
reachable shelter instead of being split, regardless of model compliance.

**Phase A — real transit delay.** The biggest structural change. Two
things were true before this phase that aren't obvious in isolation but
matter a lot together: `Transport.busy_until` existed as a field but was
never written anywhere, and `DispatcherAgent`'s output was computed every
timestep and then **discarded** — stock deduction and delivery both
happened directly off `DepotAgent`'s allocation, instantly, regardless of
what the Dispatcher decided. Fixed by:
  - New `Shipment` tracking in `world.py`: a dispatched delivery gets a
    real `arrival_timestep` (grid distance ÷ vehicle speed), and the
    vehicle is marked busy for a full round trip.
  - `DispatcherAgent` rewritten to actually control outcomes: called once
    per depot (fixing a second pre-existing bug — multi-depot allocations
    to the same shelter were silently colliding in a shared dict before
    this), given real depot/shelter positions, and its vehicle assignments
    now determine what's actually shipped. Only the shipped portion leaves
    depot stock; whatever doesn't get assigned to a vehicle stays put and
    is reconsidered next round.
  - `unmet_urgent_need` is now scored against **arrived** shipments, not
    the moment of negotiation.

Three more bugs surfaced and were fixed during Phase A's testing, all
worth recording since two are about tooling correctness rather than the
simulation itself:

1. **Parallel field-report parsing had a genuine determinism bug.**
   `run_real_disaster.py`'s concurrent parsing collected results in
   thread-completion order, not shelter order. Harmless before Phase A
   (Dispatcher output was discarded), but with dispatch now consequential,
   this order could influence which deliveries actually shipped when the
   fleet couldn't cover everyone. Confirmed directly: parallel and serial
   modes gave different `unmet_need` with an identical seed before the
   fix, byte-identical after it (rebuilding the results dict in canonical
   shelter order).
2. **The local model echoed the Dispatcher prompt's placeholder key
   literally.** The prompt's example used `{"vehicle_id": [...]}` meaning
   "a vehicle ID goes here"; the local model read it as literal text and
   used the string `"vehicle_id"` as an actual key — twice, in one
   response, which JSON's duplicate-key semantics silently resolved by
   keeping only the (invalid) second entry. Caught via a dedicated
   real-model debug trace (`run_shipment_trace.py --debug`) showing
   `dispatched_this_step=0.0` at every timestep despite `DepotAgent`
   producing valid allocations. Fixed with an unambiguous example using
   real-format IDs (`"T1"`, `"T2"`) and an explicit instruction never to
   use the literal word as a key, plus a defensive rejection of that exact
   string in code.
3. **Truncation.** The Dispatcher's `max_tokens=500` was too tight for the
   larger multi-vehicle structure Phase A introduced — confirmed in the
   same debug trace (`"resource":` cut off mid-object). Raised to 900.

After both fixes, a real-model trace was verified line-by-line against
hand-computed ground truth (grid distance ÷ vehicle speed for every
depot-shelter pair): every arrival timestep, every `busy_until` value, and
every vehicle-availability window matched exactly, including the local
model repeatedly hallucinating an unavailable vehicle ID and the defensive
check correctly rejecting it every time.

---

## Dispatch-Bottleneck Discovery

With Phase A working correctly, a re-run of the real-disaster comparison
produced a new, cleaner mystery: `local_everywhere` and
`local_parsing_haiku_negotiation` converged to **exactly** the same +7.7%
gap, with nearly identical `cumulative_unmet` at every timestep — despite
one config negotiating with Haiku and the other with the local model.

**Hypothesis:** `local_parsing_haiku_negotiation`'s dispatcher is still
`local` — `MODEL_CONFIGS["local_parsing_haiku_negotiation"] = ("local",
"claude-haiku...", "local")`. Before Phase A this didn't matter (dispatch
was decorative). Now, if the local dispatcher is the real bottleneck, a
better negotiator upstream could be getting capped by a weaker dispatcher
downstream regardless of how good its decisions were.

A dedicated diagnostic (`run_dispatch_bottleneck_diagnostics.py`) confirmed
this directly: only **~14-15% of what either config's `DepotAgent` decided
to allocate actually made it onto a vehicle**, and vehicle utilization was
low for both (15.3% and 6.9%) rather than high — meaning available
vehicles were sitting idle, not maxed out. That's a decision-quality
bottleneck, not a capacity shortage, and it's shared by both configs
regardless of negotiation quality.

**Fix:** added a new config, `local_parsing_haiku_negotiation_dispatch`
(Haiku for both negotiation and dispatch, local only for field-report
parsing), to test whether promoting dispatch recovers the negotiation
advantage the local dispatcher was masking.

---

## Real-Data Validation: The Palisades Fire (January 2025)

The real-disaster scenario above scaled a combined Palisades+Eaton
narrative synthetically (population surge factors, arbitrary grid
distances). To make a defensible claim about whether this system could
plausibly have helped route real aid, a second scenario was built instead
from actual, named, sourced locations and distances specific to the
**Palisades Fire only** (not combined with Eaton) — see
`run_palisades_scenario.py`'s module docstring for full sourcing.

### What's real vs. estimated

| Element | Status | Detail |
|---|---|---|
| 3 shelters | **Real** | Westwood Recreation Center, El Camino Real Charter HS, Pasadena Convention Center — the only 3 confirmed overnight shelters for Palisades evacuees (a 4th candidate, Cheviot Hills Recreation Center, was excluded: its real role was missing-persons/reunification, a different function, not padding for a bigger number) |
| 2 depots | **Real** | UCLA Research Park West (real Disaster Resource Center) and Malibu Pier staging area |
| Grid positions & distances | **Real**, derived | Equirectangular projection of approximate real coordinates (2 mi/grid-unit), preserving actual relative distances — e.g. Westwood↔UCLA are genuinely ~1.3 miles apart and land 1 grid-unit apart |
| Fire epicenter | **Real**, approximate | Placed near the actual ignition point in the Santa Monica Mountains |
| Timesteps | **Real** | 24, matching the actual containment window (Jan 7-31) |
| Fleet size (5) | **Estimated** | No sourced relief-vehicle count exists (firefighting apparatus counts are documented but are a different fleet entirely) |
| Shelter population (150 each) | **Estimated** | Real reporting gives ~450 combined peak overnight occupancy with no per-shelter breakdown; split evenly as the simplest defensible assumption |
| Depot stock | **Estimated** | Left at the model's natural random default rather than an artificial scarcity multiplier — with population now realistically small, no manufactured scarcity was needed to get a meaningful result |
| Wildfire spread rate (0.5/timestep) | **Approximated, with a documented tradeoff** | See below |

One fabricated source was identified and discarded during research: a
page listing named "Pacific Palisades Community Shelter" locations with
555-prefix phone numbers and sequential fake addresses — a giveaway of
fabricated content, not a reporting error worth citing.

### The constant-rate vs. front-loaded-growth tradeoff

The real fire's growth was extremely front-loaded (10 acres to 200 acres
in 20 minutes; tens of thousands of acres within 48 hours), but the world
model only supports a constant spread rate. A rate fast enough to match
how quickly the real fire threatened Westwood/UCLA/Malibu (hours to days)
would also, by around day 13 of 24, reach Pasadena Convention Center's
grid distance from the epicenter — incorrectly implying the Palisades
fire threatened Pasadena, which it didn't (Pasadena was hit by the
separate, simultaneous Eaton Fire, not modeled here). The slower rate that
keeps Pasadena's distance unreached for the full 24 days was chosen
instead, accepting an understated early-threat timeline as the more
honest tradeoff: a wrong geographic claim would undermine the scenario's
credibility more than an understated pace would.

### Results (6 repeats per config, pooled across two independent runs)

A second independent run (3 more repeats per config) was done specifically
to check whether the first run's ranking held up — it didn't, in one
important respect, which is exactly why the check mattered.

| config | run 1 (3 reps) | run 2 (3 reps) | pooled (6 reps) | cost/run |
|---|---|---|---|---|
| local_everywhere | +21.5% ± 0.0 | +21.5% ± 0.0 | +21.5% ± 0.0 | $0.000 |
| local_parsing_haiku_negotiation | +20.5% ± 0.1 | +20.8% ± 0.2 | +20.65% ± 0.15 | $0.065 |
| haiku_everywhere | +19.3% ± 0.4 | +19.2% ± 0.6 | +19.27% ± 0.45 | $0.159 |
| local_parsing_haiku_negotiation_dispatch | +18.9% ± 0.5 | +19.3% ± 0.3 | +19.07% ± 0.42 | $0.134 |

`local_everywhere` reproduced exactly (deterministic, no surprise). The
important thing: **the ranking between `haiku_everywhere` and
`local_parsing_haiku_negotiation_dispatch` flipped between the two runs**
(dispatch-upgraded mixed ahead in run 1, `haiku_everywhere` very slightly
ahead in run 2). The original write-up of this result claimed the mixed
config "edges out" full Haiku — that was an overclaim, caught by simply
re-running it. With std this size on only 3 (now 6) repeats, these two
configs are **statistically indistinguishable from each other**, not one
beating the other.

**The defensible claim is "matches full Haiku quality at ~83% of the
cost," not "beats it."** What replicated solidly across both runs, and is
the actually robust part of this finding: `local_parsing_haiku_negotiation`
(local dispatch) stayed clearly and consistently worse than both
Haiku-dispatch configs in *every* repeat of *both* runs — the
dispatch-bottleneck finding holds up under replication even though the
"beats vs. matches" nuance didn't.

Depot stock in this scenario covers ~18.8 timesteps of demand (abundant,
not scarce) — unlike the earlier synthetic scaling, difficulty here comes
from real route disruption as the fire spreads, not manufactured scarcity.
That every config, including the completely free `local_everywhere`,
lands within ~21.5% of the theoretical best-possible allocation on a
scenario built from the real Palisades Fire's actual shelter network and
timeline is a defensible basis for the claim this system could plausibly
have provided meaningfully efficient routing support — not "would have
solved the disaster," but "would have gotten allocation decisions
respectably close to optimal, cheaply, working only from noisy reports."

---

## Phase D: Lookahead/Reserve Logic

`DepotAgent` now optionally receives `lookahead_context`: exact resupply
timing (timesteps until next convoy, expected resupply fraction — real
operational facts a coordinator would know, not noisy) plus a rolling
3-round history of average shelter urgency, so it can reason about whether
to spend freely now or hold back for an anticipated worse round.

Tested directly (`run_lookahead_test.py`): held stock, shelter requests,
and route status identical across two cases — "resupply imminent, urgency
stable" vs. "resupply far away, urgency rising sharply" — and compared
allocation behavior. All three configs tested, including `local_everywhere`,
showed the correct direction (more allocated in the first case than the
second), across every repeat. One caveat worth recording: `haiku_everywhere`
and `local_parsing_haiku_negotiation_dispatch` (same underlying model)
showed a real magnitude difference between two separate 3-repeat runs
(+7.0 vs. +2.33 delta) — Haiku's own sampling variance is large enough that
a precise magnitude comparison between local and Haiku isn't solid yet,
though the qualitative finding (lookahead context is being used, not
ignored, by every tier including local) held up cleanly.

---

## Phase C: Inter-Depot Coordination

New `InterDepotCoordinatorAgent`: a single-purpose agent, called once per
timestep (not per depot), deciding whether any depot should transfer
surplus stock to another. Physically identical mechanics to a Phase A
shipment (`Transfer`/`dispatch_transfer`/`resolve_transfer_arrivals` in
`world.py`) — real travel time, and the transfer consumes a vehicle from
the same shared fleet shelter deliveries use, so it's a genuine tradeoff,
not a free side-channel. Verified directly: watched vehicle availability
shrink from 5→3→2→1→1 across a real run as transfers and shelter
deliveries competed for the same trucks, and confirmed the isolated
dispatch/arrival math exactly (distance 6, speed 2 → arrives at t=3, stock
moves exactly the dispatched quantity).

**Decision quality testing (`run_interdepot_test.py`) surfaced the third
independent confirmation of a pattern this project has now seen three
times.** Two controlled cases: a clear surplus/deficit scenario (should
transfer) and a balanced scenario (should not). Haiku scored 100% correct
on both, every repeat. Local's first result looked like appropriate
caution (0% transfer rate on both cases) — but debugging the raw output
showed this was actually a hallucinated vehicle ID (echoing this project's
own prompt example, `"T3"`, rather than a real ID from the given list) being
correctly rejected by defensive validation, not a real judgment. After
fixing the prompt (same fix pattern as the Dispatcher's placeholder-echo
bug in Phase A: concrete example using a real-format ID, explicit
anti-hallucination instruction), local's true behavior emerged: **100%
transfer rate on BOTH cases, with byte-identical output regardless of
which scenario it was given.** Not reasoning to a wrong conclusion —
not engaging with the input content at all.

### Recommended architecture (established across four independent findings)

| Role | Task type | Local model | Verdict |
|---|---|---|---|
| Field-report parsing | Extraction (single-entity, mechanical) | Reliable | Local is fine |
| Depot negotiation | Reasoning (weigh competing claims) | Measurably worse quality | Use Haiku |
| Dispatch | Reasoning (vehicle/capacity tradeoffs) | Discarded >85% of negotiation signal | Use Haiku |
| Inter-depot coordination | Reasoning (compare entities, judgment call) | Ignored input content entirely | Use Haiku |
| Risk-aware negotiation (Phase R, below) | Reasoning (weigh urgency against real danger) | Actively anti-correlated with sensible judgment | Use Haiku |

**The pattern is consistent enough across four separate, independently-discovered
roles to state as a general principle, not a per-role coincidence: local
models in this project are reliable for extraction but not for reasoning
tasks that require weighing multiple entities against each other under a
tradeoff.** `local_parsing_haiku_negotiation_dispatch` — local only for
field-report parsing, Haiku for everything else — is the config that
actually follows this principle (`InterDepotCoordinatorAgent` reuses
`depot_model`, so coordination is already Haiku here too, with no separate
change needed). `local_everywhere` is retained in `MODEL_CONFIGS` only as
the historical/diagnostic baseline every quality finding above was
measured against — not a recommended deployment choice for any scenario
using 2+ depots, real dispatch decisions, or Phase R risk-taking.

---

## Phase D/C Aren't Universally Beneficial: an Ablation Result on the Palisades Scenario

After Phase D and C were wired into `run_palisades_scenario.py`, a fresh
run of the real-data scenario (`local_everywhere`, `haiku_everywhere`,
`local_parsing_haiku_negotiation_dispatch`, 3 repeats each) showed every
config getting slightly *worse* and the two Haiku configs getting
meaningfully more expensive than the pre-D/C pooled numbers above:

| config | pre-D/C (pooled, 6 reps) | post-D/C (3 reps) | delta |
|---|---|---|---|
| local_everywhere | +21.5% | +22.1% | +0.6pp |
| haiku_everywhere | +19.27% | +19.8% | +0.5pp (cost +~10%) |
| local_parsing_haiku_negotiation_dispatch | +19.07% | +19.3% | +0.2pp (cost +~15%) |

All three configs moved the same direction, which is more suggestive than
pure noise, but not proof on its own. Rather than accept "D/C made things
worse" as a conclusion from a correlation, this was tested directly with
an ablation: same config, same seed, 3 repeats each, with `--disable-lookahead
--disable-interdepot` toggling Phase D and C off entirely.

| | gap % | cost/run |
|---|---|---|
| Ablated (D+C disabled) | **+18.7% ± 0.4** | $0.146 |
| Full (D+C enabled) | **+19.4% ± 0.2** | $0.155 |

**`transfers=0` in every single repeat of the full-pipeline run.** Phase C
never fired at all in this scenario — direct confirmation of what the
scenario's own numbers already predicted: starting stock covers ~18.8
timesteps of demand (abundant, not scarce), so there was never a genuine
surplus/deficit gap between the two depots worth transferring over. Its
cost here is pure overhead: one wasted coordination call every timestep a
vehicle happens to be free, correctly deciding nothing should be done,
but paying for the call regardless.

The ablated run's gap is not just cheaper but genuinely *better*
(+18.7% vs. +19.4%, outside the overlapping std ranges) — pointing at
Phase D specifically, not just C, as mildly counterproductive here. Giving
`DepotAgent` resupply-timing and urgency-trend context it doesn't need in
an abundant-stock scenario appears to nudge it toward occasional
unnecessary caution (holding back stock a shelter could use immediately),
rather than being harmless extra information it simply ignores.

**Conclusion: Phase C and D are targeted tools for scarcity and cross-depot
imbalance, not universally beneficial additions.** Applying them to a
well-supplied, single-fire-front scenario like Palisades adds cost
(confirmed: coordinator calls with nothing to decide) and can mildly hurt
quality (confirmed: D's context biasing toward reserve-holding when there's
no real resupply risk to hedge against). The Palisades scenario's actual
difficulty is route disruption from the spreading fire, not resource
scarcity — which D and C don't address at all. This doesn't invalidate
either phase; it means their value is scenario-dependent, and a real
deployment should probably detect (or be told) whether a scenario has
genuine multi-depot imbalance or supply-timing risk before paying their
cost. `run_palisades_scenario.py --disable-lookahead --disable-interdepot`
is the tool for checking that on any future scenario.

---

## Phase R: Risk-Tolerant Routing

Prompted by a direct observation while reviewing the dashboard: routes were
visually crossing the fire circle without being flagged blocked (this led
to the route-blocking geometry fix documented under "Issues found" below),
which raised a legitimate follow-up question — shouldn't a real coordinator
sometimes take a calculated risk and push aid through danger for a
critical shelter, rather than treat every non-hard-blocked route as
equally safe?

**Design.** Hard-blocked routes (literally inside the fire radius) stay a
permanent, binary wall, unchanged. A new "risk halo" beyond the fire's edge
(`WILDFIRE_RISK_BUFFER = 3.0` grid units) gets a continuous risk score,
sampled along the actual route path, falling linearly from 1.0 at the fire
edge to 0.0 at the buffer's outer boundary (`route_risk_score()` in
`world.py`). Attempting a risky route is a real gamble, resolved by the
simulation, not the LLM (`resolve_risk_outcome()`): success, partial cargo
loss, or total loss of both cargo and vehicle (`Transport.destroyed`,
permanent for the rest of the run). All three reasoning agents
(`DepotAgent`, `DispatcherAgent`, `InterDepotCoordinatorAgent`) receive
risk scores as real context and are told explicitly to weigh it against
urgency/population — gamble for a critical shelter, don't for a marginal
one. The optimizer baseline discounts risky routes by their *exact*
expected surviving fraction (`1 - 0.5*risk`, verified empirically against
200,000 simulated outcomes to 4 decimal places) rather than pretending it
can gamble the way the multi-agent system now genuinely can.

### Decision-quality testing (`run_risk_test.py`)

Two controlled cases, isolating `DepotAgent`: a shelter on a risky route
(risk=0.6) that's either critical (urgency 0.9, Case A — worth the risk)
or low-priority (urgency 0.15, Case B — not worth it), with a safe
alternative shelter present as a genuine choice, not a strawman.

| config | S1 (risky) Case A | S1 (risky) Case B | delta |
|---|---|---|---|
| haiku_everywhere | 38.60 | 6.00 | **+32.60** |
| local_parsing_haiku_negotiation_dispatch | 37.00 | 6.00 | **+31.00** |
| local_everywhere | 21.00 | 30.00 | **-9.00** |

Haiku (both configs sharing that model) got this right, consistently
across every repeat: commit real resources to the risky shelter when it's
critical, redirect to the safe alternative when it isn't.
**`local_everywhere` got the direction backwards** — it sent *more* to the
risky shelter when it was *less* urgent. This isn't "ignores risk" or
"ignores urgency" (either of which would show as a near-zero delta) — it's
actively anti-correlated with a sensible read of the situation, a more
specific and more concerning failure than the pattern seen in negotiation
quality, the dispatch bottleneck, or inter-depot coordination.

### Full-loop confirmation on the Palisades scenario

Even with a stubbed model, a single real-geometry run showed Phase R
engaging often, not as a rare edge case: 8 risky attempts, 1 vehicle lost,
in 24 timesteps. With real models (3 repeats each):

| config | pre-Phase-R gap | post-Phase-R gap | risky attempts | vehicles lost |
|---|---|---|---|---|
| local_everywhere | +21.6% | **+25.2%** (worse, deterministic) | 8 | **3 (37.5% loss rate)** |
| haiku_everywhere | +18.1% | +18.1% (same mean, variance +0.6→+1.4 std) | 3-5 | 0-1 |
| local_parsing_haiku_negotiation_dispatch | +18.9% | +18.3% (slightly better) | 1-2 | 0 |

**`local_everywhere`'s isolated test failure translated directly into a
real, quantified cost**: 3 outright vehicle losses and 2 partial losses
out of 8 gambles — a genuinely bad decision policy burning real resources
in a real scenario, not just a wrong answer on a synthetic test.
**Haiku's variance increase is a feature, not a bug**: repeat 3 hit a real
vehicle loss and landed at +19.7% against the other two repeats' +17.2-
17.3% — sensible risk-taking can still have a bad outcome sometimes, and
the simulation now captures that instead of treating every non-blocked
route as a guaranteed success. The mixed config took fewer gambles (1-2
vs. Haiku's 3-5) with zero losses, likely reflecting local's noisier
field-report extraction feeding a slightly different urgency signal into
an otherwise-identical negotiator, not a difference in the negotiator's
own judgment.

**This is the fourth independent role — after negotiation, dispatch, and
inter-depot coordination — where local proves unsuitable for a multi-factor
reasoning task while remaining fine for extraction.** Folded into the same
architectural principle below rather than treated as a separate caveat,
since it's the same underlying fact showing up again:
`local_parsing_haiku_negotiation_dispatch` already covers this correctly,
since risk-aware negotiation runs on Haiku there like every other reasoning
role.

---

## Fire Radius Realism Fix

Caught by building the Leaflet/real-map version of the dashboard and
actually looking at it: the fire radius circle, drawn on real geography
for the first time, visibly covered most of Santa Monica Bay (open ocean)
and reached from Simi Valley to past Pasadena by day 24 -- a circle about
50x larger than the real fire's actual footprint. The abstract grid view
used everywhere before this had completely hidden the problem; a real map
is what made it obvious.

**Root cause:** `wildfire_spread_rate=0.5` grid-units/timestep was chosen
for exactly one purpose -- keeping Pasadena's distance from the epicenter
unreached through day 24 (documented in the scenario's original
calibration notes). Nobody checked whether the circle's *absolute* size
was realistic at any point along the way. It wasn't: by day 24, the old
rate produced a 12-grid-unit (24 mile) radius, ~1,800 sq mi of "fire" --
against a real documented footprint of 23,448 acres (36.6 sq mi), about
1/50th the size. A circular danger zone also has no concept of water or
terrain, so it "burned" the Pacific Ocean without anything flagging that
as absurd.

**Fix:** recalibrated `wildfire_spread_rate` to 0.0711 grid-units/timestep,
derived directly from the real footprint: 23,448 acres treated as an
equivalent circle gives radius = sqrt(36.6/pi) ≈ 3.41 mi ≈ 1.71 grid-units;
reaching that by day 24 (full containment) gives 1.71/24 ≈ 0.0711/timestep.

**The natural worry going in: would a realistically-sized fire ever come
close enough to matter, given every real location here sits 8-26 miles
from the epicenter?** Checked directly rather than assumed -- yes,
meaningfully so, and in a more interesting way than expected. Depots and
shelters sit in different directions *around* the epicenter, not
clustered on one far side of it, so several straight-line paths between
them still pass close to the center even though no single endpoint does.
Of the 7 real routes: 3 eventually become hard-blocked (day 16-24 by the
actual simulation code, vs. day 3-11 under the old miscalibration), 3 stay
risky-but-passable for the entire 24 days, and exactly one (D1-S3,
UCLA↔Pasadena) is never affected at all -- correctly matching the
real-world fact that Pasadena was Eaton Fire territory, not Palisades.
**Every shelter keeps at least one viable, if sometimes risky, route to
at least one depot for the full 24 days** -- the total shelter isolation
the old miscalibration produced by day 11 does not happen once the fire
is sized correctly. Verified against the actual `advance_disaster()` code
directly, not just a standalone geometry script.

This is a better outcome than either "fire this size never affects
anything" (the failure mode initially worried about) or "everything gets
cut off" (what the old, oversized model actually produced) -- a
realistically-sized fire still creates real, meaningful route risk and
some permanent disruption, just later in the timeline and via a
geometrically honest mechanism, rather than an absurdly large circle that
happened to produce dramatic-looking but physically nonsensical results.

**Consequence for prior results:** the Palisades scenario numbers and
dashboard discussed in every section above this one (Real-Data Validation,
Phase D/C ablation, Phase R full-loop test, the "total isolation by day
11" finding) were all generated under the old, oversized fire model. They
should be treated as historical/pre-fix from here forward -- a fresh run
under the corrected calibration is needed before any of those specific
numbers can be cited as current.

---

## RNG Stream Contamination: A Confirmed Bug Invalidating Every Phase R+ Config Comparison

Surfaced while investigating a genuinely strange result after the Fire
Radius Realism Fix: `haiku_everywhere` was shipping far more total volume
than `local_everywhere` (334.8 vs. 39.4 units in one comparison, 8x more,
across 24 vs. 7 shipment events) yet ending up with *worse* absolute
unmet-need. Volume, timing, and resource-type mix were all checked
directly and all favored Haiku -- none of them explained the result.

**Root cause, found by adding arrival/shelter-stock logging and checking
directly:** shelter *population* -- meant to be ground truth, identical
across configs regardless of which model is negotiating -- was diverging
significantly between configs. One direct comparison: `local_everywhere`'s
S1 population grew 150→161 by day 20; `haiku_everywhere`'s grew 150→221
over the same window, 37% higher, purely from a difference in which
config was running. Since the final unmet-need formula multiplies
shortfall by `population/100`, a config that happens to draw a higher
population trajectory faces a bigger penalty for the identical underlying
shortfall -- contaminating every comparison, not reflecting genuine
allocation quality.

**The mechanism:** population growth in `advance_disaster()` draws from
`rng.random()` / `rng.randint()`. Phase R's `resolve_risk_outcome()` --
called only when a risky shipment or transfer is actually attempted --
drew from the exact same shared `rng` object. Since `haiku_everywhere`
attempts far more shipments than `local_everywhere` (confirmed: 24 vs. 7
events in the run that surfaced this), it also makes more
`resolve_risk_outcome()` calls, consuming a different number of draws
from the shared stream. That silently desyncs the RNG's internal position
between configs from the point their behavior first differs onward --
every subsequent "identical seed" draw (population growth, and anything
else sharing that stream) produces different values per config, even
though both started from the same seed. **Every config comparison since
Phase R was introduced had configs technically facing subtly different
underlying scenarios, not the same scenario under different agent
policies** -- undermining the controlled-experiment premise behind every
side-by-side result in this project since that point.

**Fix:** a dedicated `risk_rng`, separate from the `rng` driving world-
ground-truth randomness (population growth, disaster progression,
situation-report noise), derived once at the very start of each trial --
`risk_rng = random.Random(rng.random())` -- before any config-dependent
behavior can diverge. From that point, `rng` drives only world state and
`risk_rng` drives only risk outcomes; different risk-attempt counts
between configs now only affect risk *outcomes*, never the underlying
scenario. Applied to `apply_shipment_with_risk()`/`apply_transfer_with_risk()`
(the shared helpers in `run_simulation.py`) and every one of the 6
orchestration scripts that calls them.

**Verified directly, not just assumed fixed:** constructed two runs with
the same seed and same world, one attempting 0 risky shipments per day and
one attempting 5 (75 extra `risk_rng` draws total across 15 days) --
population trajectories came out byte-for-byte identical between them
after the fix, confirming different risk-attempt counts no longer affect
shared world state at all.

**Consequence:** every Palisades number discussed anywhere above this
section (and the Phase D/C ablation, run under a different fire model but
the same contaminated RNG sharing) needs re-generating under this fix
before any specific number can be trusted. This is now the second
"needs a fresh run before trusting these numbers" flag stacked on top of
the Fire Radius fix above -- both should be re-run together, not
separately, to avoid a third round of partially-stale results.

This also means the original mystery that prompted this investigation --
why does shipping more lead to worse outcomes? -- may turn out to have TWO
contributing causes, only one of which is now fixed: this RNG
contamination (fixed), and a possible separate quirk in how
`unmet_urgent_need` evaluates a fixed daily target with no rollover credit
(flagged, not yet confirmed or fixed -- see Open Items). Re-running under
the RNG fix is necessary to see how much of the original puzzle remains
once ground truth is actually held constant across configs.

### Resolution: re-run confirms the RNG bug was the entire explanation

Re-ran all three configs under both fixes together (corrected fire radius
+ RNG separation). Two things to check, both confirmed directly:

1. **Population is now byte-for-byte identical across all three configs,
   every single day** (checked at days 1, 5, 10, 15, 20, 24) -- direct
   proof the fix works in a real run, not just the isolated 0-vs-5-attempts
   test above.
2. **The volume/outcome relationship now makes sense.** With ground truth
   actually held constant:

   | config | shipped | arrived | events | final unmet |
   |---|---|---|---|---|
   | local_everywhere | 68.1 | 68.1 | 7 | 26,718.2 (worst) |
   | haiku_everywhere | 372.2 | 334.6 | 21 | 25,878.0 |
   | local_parsing_haiku_negotiation_dispatch | 435.2 | 429.5 | 26 | 25,485.6 (best) |

   Clean and monotonic: more shipped, more arrived, better outcome. The
   headline ranking also now makes sense for the first time in this whole
   investigation -- `local_everywhere` worst (+72.5%), `haiku_everywhere`
   next (+68.6%), and the recommended mixed config best (+65.2%) at 85% of
   full Haiku's cost, consistent with every other finding in this project
   about the mixed config's value proposition.

**The RNG bug was the entire explanation.** The second, unconfirmed
suspicion about `unmet_urgent_need`'s no-rollover-credit design doesn't
appear to be needed to explain this data -- volume and outcome track
sensibly once the shared-stream contamination is removed. That open
question (marked in Open Items) can be downgraded from "actively
suspected" to "a theoretical modeling quirk worth knowing about, not
something currently distorting results" unless a future scenario
surfaces it again independently.

---

## Depot Processing Order Fleet-Starvation Bug

Caught by direct observation of a real run's dashboard: in the Palisades
scenario, D2 (Malibu Pier) was shipping *zero* units across the entire
24-day window in both Haiku-negotiation configs, while D1 (UCLA) handled
essentially all delivery volume and S1 (Westwood) alone absorbed 67-76%
of everything shipped. Verified directly rather than assumed a data quirk:
D2-S2's route was never even hard-blocked (open the full 24 days), and
D2's stock nearly *tripled* over the run (181→544 food) purely from
resupply, since nothing was ever shipped out of it.

**Root cause, confirmed via direct trace:** the 5 transports are a single
shared fleet across both depots, and `world.depots.values()` always
yields the same fixed order (D1, then D2) every timestep. Whichever depot
is processed first gets first claim on whatever vehicles happen to be
free that day. A real-model trace showed **D2 seeing 0 available vehicles
on every single day across the first 10 days** -- including days where
D1 saw 3-5 free vehicles and simply took what it wanted. Not a decision-
quality issue on D2's part; by the time D2 even got a turn, the pool was
already empty. This also explained an item that had been sitting
unexplained in Open Items: the mixed config attempting fewer risky
shipments than `haiku_everywhere` despite sharing the same negotiation
model -- never actually about differing judgment, just D1 monopolizing
the fleet regardless of which config was running.

### First fix attempt: simple alternation -- confirmed insufficient, not assumed sufficient

The first fix rotated depot processing order by timestep parity (D1
first on even days, D2 first on odd days, or vice versa). Re-ran the
exact same trace that caught the original bug rather than declaring
victory on the logic alone: **D2 still saw 0 available vehicles on every
single day**, even on days it was now processed first. The alternation
period (2) resonated with the vehicle round-trip cycle (~2 days in this
scenario): D1 claimed newly-freed vehicles on its "first" days, they
weren't back yet on D2's very next "first" day, and by the time they
returned, priority had already cycled back to D1. A fixed schedule can
coincidentally align with a fixed physical cycle and fail to redistribute
anything, even though the code "looks" fair.

### Working fix: fair-share ordering by cumulative outcome, not by turn schedule

Replaced the fixed-schedule approach with one that tracks actual cumulative
vehicle claims per depot (`world.depot_vehicle_claims`, incremented in
`dispatch_shipment`/`dispatch_transfer`) and always processes whichever
depot has been served LEAST so far. This can't resonate with any periodic
cycle because it responds to real history, not a predetermined pattern.
Re-verified with the identical adversarial trace: D2 now gets genuine
priority on Day 3 (3 vehicles) and Day 11 (4 vehicles) -- the exact
moments vehicles became newly available while D2 was still behind on
cumulative access -- something that never happened under either the
original bug or the failed alternation attempt.

**Methodological note worth keeping:** the failed first attempt is left
in this writeup deliberately. It would have been easy to ship the
alternation fix, see the code "looks" more fair, and move on without
re-running the verification trace -- exactly the kind of plausible-looking
wrong answer this project has run into more than once (the ablation
result, the RNG contamination, both looked fine until directly checked).
Re-testing with the same adversarial case that caught the original bug is
what caught the first fix's failure.

---

## Issues found and fixed along the way

(Issues #5 and #6 — the `DepotAgent` clamp-order bug and the scarcity-ceiling
finding — are documented above in "Real-Disaster Stress Test," since they
only surfaced during that test and are easier to follow alongside the
numbers that revealed them. Issues from Phase E/B/A and the dispatch-
bottleneck discovery are documented in their own sections above for the
same reason.)

### 1. Wildfire caused an unrecoverable depot stockout (fixed: periodic resupply)

The original model gave each depot a single, one-time stock allocation with
no resupply mechanism. Under wildfire's faster evacuee influx, every
config's depot ran completely dry by roughly timestep 9 of 15 — after that,
`max_shelter_urgency` was pinned at 1.0 and unmet-need accumulated every
remaining timestep with no config able to do anything differently. Most of
wildfire's original cumulative unmet-need, and therefore most of the
apparent config-vs-optimizer gap, was being generated in a regime no
negotiation strategy could affect.

**Fix:** depots now receive periodic resupply convoys (a fraction of their
*original* stock, on a fixed schedule). Wildfire-specific: a convoy can't
reach a depot inside the current fire radius — rather than leaving that
depot permanently cut off (which recreated the same unrecoverable collapse,
since fire radius only grows), a blocked depot instead gets a reduced-rate
emergency airdrop. Wildfire stays meaningfully harder than the other two
types, but no depot is left with zero recourse.

### 2. Uniform resupply parameters distorted flood's percentage-gap metric

Applying wildfire's resupply settings (50% of stock every 5 timesteps)
uniformly to all three disaster types dropped flood's optimizer baseline
from 156.5 to 41.97 — a 73% reduction. Every config's absolute unmet-need on
flood was actually *lower* than the old baseline, but because the
denominator collapsed, gaps inflated to +330%, +104%, +81% — a
division-by-small-number artifact, not a real finding.

**Fix:** resupply parameters are now tuned per disaster type instead of
applied uniformly. Flood's actual pre-fix problem was much milder than
wildfire's (4/15 timesteps at near-zero stock vs. total, permanent
collapse), so it now gets a single smaller top-up (15% of stock at t=10)
rather than wildfire's two large ones (50% at t=5 and t=10). This brought
flood's baseline to a defensible ~104 — a real improvement over the
original 156.5 without manufacturing an artificially generous denominator.

### 3. `run_sweep.py` now reports absolute unmet-need alongside percentage gap

Every per-scenario line, pooled summary, and CSV now carries
`unmet_need_mean`/`std` next to `optimality_gap_pct`, plus the optimizer's
own baseline printed for reference and a standing reminder to sanity-check
percentage gap against the absolute number. This is what caught issue #2
above, and should catch any future denominator problem immediately instead
of requiring a full investigation to notice.

### 4. Route-blindness hypothesis (tested, falsified)

Early hypothesis for wildfire's anomalous results: `DepotAgent.allocate()`
isn't given route-blockage information, so it might keep allocating stock
to shelters whose route the fire had just cut off — stock deducted but
never deliverable. A dedicated diagnostic (`run_wildfire_diagnostics.py`)
measured this directly as `wasted_stock_this_step` and found it at exactly
0.0 across every config, including ones with unrelated truncation issues.
Route-blockage in the current model is checked as an exact grid-coordinate
match against fixed depot/shelter positions, which rarely coincides with
the small number of actual depot-shelter edges — so this mechanism, while
plausible in principle, wasn't what was actually happening. The real driver
turned out to be the resupply/depletion issue documented above.

---

## Files

- `simulation/world.py` — three disaster-type damage models, periodic
  resupply with per-type tuning and wildfire airdrop fallback,
  `RESOURCE_CRITICALITY` weights (Phase E), `Shipment`/`dispatch_shipment`/
  `resolve_arrivals` for real transit delay (Phase A).
- `optimizer/baseline.py` — LP objective and `unmet_urgent_need` now both
  population- and criticality-weighted, consistently (Phase E fix).
- `agents/field_report_agent.py` — now also extracts a noisy population
  estimate from situation-report text (Phase E).
- `agents/depot_agent.py` — allocation clamp scales proportionally under
  scarcity instead of sequentially by dict order (Issue #5); receives and
  enforces `route_status` per shelter (Phase B); prompt includes
  criticality weights and population context (Phase E).
- `agents/dispatcher_agent.py` — rewritten for Phase A: called once per
  depot, given real positions, its vehicle assignments now actually
  control what ships; fixed the placeholder-echo and truncation bugs
  found via real-model testing.
- `run_sweep.py` — multi-scenario, multi-repeat comparison harness;
  reports both percentage gap and absolute unmet-need. (Not yet re-run
  with the Phase E/B/A fixes — see Open Items.)
- `run_simulation.py` — `MODEL_CONFIGS` now includes
  `local_parsing_haiku_negotiation_dispatch` (Haiku on negotiation +
  dispatch, local only for parsing), added after the dispatch-bottleneck
  discovery.
- `run_wildfire_diagnostics.py` — per-timestep diagnostic trace (fire
  radius, blocked routes, resupply status, wasted stock); originally used
  to test and falsify the route-blindness hypothesis, now doubles as an
  ongoing regression check that both Phase B and Phase A's route defenses
  keep `wasted_stock` at ~0.
- `run_real_disaster.py` — real-disaster stress test calibrated to the
  Jan 2025 LA wildfires (combined Palisades+Eaton framing, synthetic
  population scaling); concurrent LLM calls for hosted-model configs
  (with a fixed determinism bug — see Phase A section), pre-flight
  stock/demand sanity check.
- `run_scarcity_diagnostics.py` — per-timestep trace of total delivered
  vs. available resource and % shelters critical, used to confirm the
  scarcity-ceiling finding (Issue #6).
- `run_shipment_trace.py` — direct, real-model trace of shipment dispatch/
  arrival timing with `--debug` raw-response visibility; the tool that
  caught both Dispatcher bugs during Phase A.
- `run_dispatch_bottleneck_diagnostics.py` — compares desired-vs-shipped
  quantities and vehicle utilization between configs; confirmed the
  dispatch-bottleneck hypothesis.
- `run_palisades_scenario.py` — the real-data-calibrated Palisades Fire
  scenario (real shelters/depots/distances/timeline, clearly-labeled
  estimates elsewhere); the most externally-defensible result in the
  project so far. Wired with Phase D lookahead, Phase C inter-depot
  coordination, and Phase R risk-tolerant routing; `--disable-lookahead`/
  `--disable-interdepot` flags allow ablation-testing D/C independently,
  which is how the "D/C aren't universally beneficial" finding above was
  confirmed rather than assumed from a correlation. Results now also
  report `n_transfers`, `n_risky_attempts`, `n_vehicles_lost`, and
  `n_partial_losses` per run; replay logs (`--replay-log`) include
  per-timestep `risk_outcomes` and `vehicles_destroyed`.
- `agents/inter_depot_agent.py` — new for Phase C: `InterDepotCoordinatorAgent`,
  reuses `depot_model` rather than being an independently tunable role;
  documents the confirmed local-model-can't-reason-about-this finding.
- `simulation/world.py` — Phase R additions: `route_risk_score()` (continuous
  risk in a buffer zone beyond the literal fire edge, sampled along the
  real route path), `resolve_risk_outcome()` (rolls success/partial-loss/
  total-loss), `Transport.destroyed` (permanent vehicle loss).
- `optimizer/baseline.py` — Phase R: risky routes discounted in the LP
  objective by their exact expected surviving fraction, verified against
  200,000 simulated outcomes to 4 decimal places.
- `run_simulation.py` — `apply_shipment_with_risk()`/`apply_transfer_with_risk()`
  now take a dedicated `risk_rng` parameter, separate from the world's main
  `rng` (RNG Stream Contamination fix, above). Every one of the 6
  orchestration scripts derives `risk_rng = random.Random(rng.random())`
  once at the very start of each trial, before any config-dependent
  behavior can diverge.
- `run_lookahead_test.py` — direct, controlled test of Phase D's
  lookahead_context (identical inputs except resupply timing/urgency
  trend, comparing resulting allocation behavior).
- `run_interdepot_test.py` — direct, controlled test of Phase C's transfer
  decisions (clear surplus/deficit case vs. balanced case), with `--debug`
  raw-response visibility; the tool that caught the coordinator's
  vehicle-ID hallucination bug.
- `run_risk_test.py` — direct, controlled test of Phase R's risk-vs-urgency
  judgment (critical shelter on a risky route vs. low-priority shelter,
  same risk, with a safe alternative present); the tool that caught
  `local_everywhere`'s wrong-direction risk-taking bug.

## Open items for a future pass

- **Highest priority (supersedes the item below): re-run the full
  Palisades scenario under the depot fleet-fairness fix, on top of the
  fire-radius and RNG fixes already applied.** D2 going from 0% to a real
  share of shipment volume will change absolute numbers and likely the
  gap%/cost comparison across all 3 configs -- this is a bigger behavioral
  change than either prior fix, since it's not a calibration correction
  but a genuine fix to which decisions the agents were even able to act
  on. Every number in this document predates this fix.
- ~~Highest priority: re-run the full Palisades scenario under BOTH fixes
  together~~ — **done, but now superseded by the item above**: population
  confirmed byte-for-byte identical across all 3 configs every day;
  volume/outcome relationship was clean and monotonic under those two
  fixes. See "Resolution" under RNG Stream Contamination above for those
  numbers -- they were correct GIVEN the fleet-starvation bug still
  present at the time, and need one more re-run now that it's fixed too.
- ~~Open question: does `unmet_urgent_need`'s no-rollover-credit design
  create a real bias against frequent-small deliveries?~~ — **downgraded**:
  the re-run above shows the volume/outcome relationship is clean once the
  RNG bug is fixed, so this doesn't appear to be actively distorting
  results. Still a theoretical modeling quirk worth knowing about
  (`shortfall = max(0, need - delivered)` really does have no memory of
  past deliveries or credit for exceeding target), but not something to
  prioritize fixing unless a future scenario surfaces the same volume/
  outcome mismatch independently of any RNG issue.
- **Re-run the toy-scale sweep (`run_sweep.py`, all 3 disaster types) with
  every fix applied** — Issue #5's clamp fix, Phase E's population/
  criticality weighting (which changes the metric's scale), Phase B's
  route-awareness, Phase A's transit delay, Phase D's lookahead context,
  Phase C's inter-depot coordination, Phase R's risk-tolerant routing, and
  the dispatch-bottleneck fix have all landed since the Headline Result
  table was generated. That table is now quite stale; the qualitative
  conclusion (mixed ≈ full-Haiku at a cost discount) has repeatedly held
  up in every re-test, but the exact percentages should be treated as
  historical, not current, until this re-run happens.
- ~~More repeats on the Palisades scenario to firm up the margin~~ —
  **done**: a second independent 3-repeat run flipped the ranking between
  `haiku_everywhere` and `local_parsing_haiku_negotiation_dispatch`,
  confirming they're statistically indistinguishable rather than one
  beating the other (see corrected Results section above). The original
  "edges out full Haiku" claim was caught and walked back specifically
  because this re-run was done.
- ~~Phases C and D from the original build-out plan~~ — **done**: both
  built, bug-hunted (a vehicle-ID hallucination in the coordinator, same
  pattern as the Dispatcher's Phase A bug), and verified against real
  models. See "Phase D" and "Phase C" sections above.
- ~~The Palisades scenario hasn't been re-run with Phase D/C wired in~~ —
  **done**: re-run and directly ablation-tested. See "Phase D/C Aren't
  Universally Beneficial" above — D and C add cost and mildly hurt quality
  in this specific scenario, since it's abundant-stock and single-fire-
  front, not scarcity- or imbalance-driven. Genuinely useful negative
  result, not a regression to fix.
- **The toy-scale sweep's flood/earthquake scenarios (which showed real
  scarcity and, for flood, an eventual optimizer-beating result) haven't
  been tested with the same D/C ablation.** Given Phase D/C's value now
  looks scenario-dependent, worth checking whether they help more in a
  scenario where scarcity or multi-depot imbalance is actually present,
  rather than concluding "D/C don't help" from Palisades alone.
- Mixed routing's higher scenario-sensitivity (22.4pp vs. Haiku's 13.5pp
  in the pre-Phase-E/B/A toy-scale sweep) isn't yet root-caused — worth a
  diagnostic pass similar to the wildfire one if it still holds after the
  sweep is re-run.
- ~~`local_parsing_haiku_negotiation_dispatch` took noticeably fewer risky
  gambles than `haiku_everywhere` despite sharing the same negotiation
  model~~ — **resolved**: this was never about differing judgment quality.
  See "Depot Processing Order Fleet-Starvation Bug" above -- D1
  monopolizing the shared vehicle fleet (now fixed) meant whichever
  config's D1 happened to claim more vehicles that run would mechanically
  show fewer D2-side risky attempts, unrelated to the negotiator itself.
- Local model's specific negotiation failure mode (dumping stock on one
  shelter? ignoring urgency signals? violating transport-capacity
  constraints?) hasn't been traced at the decision level — only the
  aggregate outcome is confirmed bad.
- Sonnet's truncation issue was worked around by exclusion, not fixed;
  revisit if there's ever a reason to bring Sonnet back into the
  comparison.