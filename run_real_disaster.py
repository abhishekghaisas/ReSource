"""
Stress-tests the validated finding (mixed-tier routing ~matches full-Haiku
quality at ~1/3 cost, see FINDINGS.md) against a scenario calibrated to an
ACTUAL disaster: the January 2025 Los Angeles wildfires (Palisades + Eaton
fires), rather than the toy-scale synthetic scenarios used for the core
validation sweep.

WHY THIS IS A DIFFERENT TEST THAN run_sweep.py's wildfire:7
-------------------------------------------------------------------------
The validated sweep used wildfire_spread_rate=0.6 (world.py's default),
n_shelters=4, 15 timesteps -- reasonable for isolating the wildfire-vs-flood
mechanism, but nowhere near the scale or speed of a real catastrophic fire.
Real numbers, sourced from Red Cross / CAL FIRE / FEMA reporting on the
Jan 2025 event:
  - Ignited under hurricane-force Santa Ana winds (gusts to 100 mph) and
    spread far faster than a typical wildfire.
  - Over 200,000 residents evacuated; 17,000+ homes destroyed/damaged.
  - Fire remained active/uncontained for 24 days.
  - Red Cross alone logged ~21,000 overnight shelter stays and 188,500
    meals across an estimated 178,000 people reached -- thin per-capita
    relief, i.e. real resource scarcity, not comfortable margins.
  - On-the-ground reports documented a norovirus outbreak in at least one
    shelter and flu/respiratory illness circulating in others once
    shelters had been crowded for a while -- a real secondary demand shock
    on top of the initial fire damage.

CALIBRATION CHOICES (each is a judgment call, documented so you can argue
with any of them):
  - wildfire_spread_rate: 1.2 (2x world.py's default 0.6) -- approximates
    wind-driven fire growing far faster than the "typical" wildfire the
    default was tuned against. This is a coarse proxy, not a physics model.
  - grid_size (14x14) and timesteps (24, ~1 day/timestep to match the
    24-day containment window) are both larger than the validated sweep's
    6x6 / 15-timestep setup, and are paired so the fire radius only reaches
    the far corners of the grid near the END of the run (~day 22) rather
    than engulfing the whole map by the midpoint -- i.e. damage escalates
    across the full 24-day event instead of front-loading.
  - population_surge_factor: 4x each shelter's baseline population, applied
    ONCE at scenario creation. An earlier version of this calibration used
    10x, which turned out to be a mistake worth documenting: at 10x, total
    starting depot stock covered only ~1.03 timesteps of total demand out of
    24 -- i.e. every config (including the optimizer) hit total resource
    exhaustion almost immediately, which reproduces the exact "unrecoverable
    stockout" pathology already diagnosed and fixed once in this project
    (see FINDINGS.md issue #1): once nobody has anything left to allocate,
    there's no decision left for a "good" vs "bad" negotiator to differ on,
    and every config's gap collapses toward the optimizer's by construction,
    not because the finding generalized. At 4x, starting stock covers ~2.6
    timesteps and resupply replaces only ~1.3 timesteps of demand per
    5-timestep interval -- a real, WIDENING deficit across the 24-day run
    (traced: optimizer's own stock depletes gradually, reaching zero around
    t=18 of 24, not t=1-2), which is both more realistic (real relief was
    strained and worsening, not instantly zero -- 188,500 meals across
    178,000 people is thin, not nonexistent) and keeps the comparison
    metric sensitive for most of the run. Depot stock itself is still
    NOT scaled up to match the surge -- real relief supply chains don't
    instantly scale either, which is the scarcity mechanism this scenario
    is meant to test.
  - disease_outbreak_timestep (10) / disease_outbreak_medical_multiplier
    (1.6x, applied once and permanent for the rest of the run) -- a stand-in
    for the documented norovirus/flu outbreaks. This is intentionally layered
    on in THIS script rather than baked into world.py's core wildfire model,
    so the validated core mechanics stay untouched -- we're testing
    robustness to an add-on shock, not silently changing the system that was
    already validated.

These are honest approximations of a real event's scale and dynamics, not a
literal reproduction -- the underlying grid/shelter/depot abstraction was
never meant to simulate 200,000 individuals one by one. Treat this as "does
the config ranking survive a realistically harsh, realistically fast,
realistically scarce version of the scenario," not as a forecast of what
would happen in an actual future LA fire.

Usage:
    # Smoke test (fast, one repeat, local_everywhere only -- no API cost)
    python run_real_disaster.py --configs local_everywhere --repeats 1

    # Real result: all three finalists, 5 repeats
    python run_real_disaster.py --repeats 5

    # Compare against a milder or more extreme version of the calibration
    python run_real_disaster.py --wildfire-spread-rate 0.8 --population-surge-factor 5
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import csv
import os
import random
import statistics
import time

from agents.base_agent import CostTracker, LLMBackend
from agents.depot_agent import DepotAgent
from agents.dispatcher_agent import DispatcherAgent
from agents.field_report_agent import FieldReportAgent
from agents.inter_depot_agent import InterDepotCoordinatorAgent
from optimizer.baseline import solve_optimal_allocation, unmet_urgent_need
from run_simulation import (MODEL_CONFIGS, URGENCY_HISTORY_LENGTH, average_urgency,
                             build_depot_summaries, compute_route_risk_for_depot,
                             apply_shipment_with_risk, apply_transfer_with_risk,
                             rotate_depots_for_fairness)
from simulation.world import (RESOURCE_TYPES, WorldState, advance_disaster, dispatch_shipment,
                               dispatch_transfer, make_scenario, resolve_arrivals,
                               resolve_transfer_arrivals, route_risk_score,
                               timesteps_until_next_resupply)

DEFAULT_CONFIGS = [
    "local_everywhere",
    "haiku_everywhere",
    "local_parsing_haiku_negotiation",
    "local_parsing_haiku_negotiation_dispatch",
]

# Reference numbers from the validated toy-scale sweep (FINDINGS.md), printed
# alongside this scenario's results so you can see the delta at a glance.
# local_parsing_haiku_negotiation_dispatch has no entry -- it didn't exist
# at toy-scale validation time (added after the Phase A dispatch-bottleneck
# diagnostic) -- TOY_SCALE_WILDFIRE_REFERENCE.get() returns None for it,
# and the summary printing handles that by printing "n/a" instead of crashing.
TOY_SCALE_WILDFIRE_REFERENCE = {
    "local_everywhere": 14.8,
    "haiku_everywhere": 6.9,
    "local_parsing_haiku_negotiation": 11.0,
}

LA_WILDFIRES_JAN2025 = {
    "description": "Calibrated to the Jan 2025 LA (Palisades + Eaton) wildfires",
    "seed": 2025,
    "n_shelters": 10,
    "n_depots": 2,
    "n_transports": 5,
    # Grid sized so max manhattan distance (26) is reached by the fire radius
    # (spread_rate * timesteps = 1.2*24 = 28.8) only near the END of the
    # 24-day run -- i.e. the fire threatens the whole simulated region by
    # about day 22, not by day 15. A smaller grid at this spread rate made
    # the entire map "on fire" by the midpoint, which front-loaded all the
    # damage and defeated the point of a 24-timestep escalating scenario.
    "grid_size": (14, 14),
    "timesteps": 24,                       # ~1 day/timestep, matches 24-day containment
    "wildfire_spread_rate": 1.2,            # 2x world.py's default 0.6
    "population_surge_factor": 4.0,         # depot stock NOT scaled -- see module docstring
    "disease_outbreak_timestep": 10,
    "disease_outbreak_medical_multiplier": 1.6,
}


def apply_population_surge(world: WorldState, factor: float) -> None:
    """
    Scale each shelter's population AND consumption_rate by `factor` (both
    together, since consumption_rate was originally derived from population --
    scaling only one would silently break the urgency/consumption model).
    Depot stock is untouched on purpose (see module docstring).
    """
    for shelter in world.shelters.values():
        shelter.population = int(shelter.population * factor)
        for r in RESOURCE_TYPES:
            shelter.consumption_rate[r] = shelter.consumption_rate.get(r, 0.0) * factor


def apply_disease_outbreak(world: WorldState, medical_multiplier: float) -> None:
    """One-time, permanent bump to medical consumption at every shelter."""
    for shelter in world.shelters.values():
        shelter.consumption_rate["medical"] = shelter.consumption_rate.get("medical", 0.0) * medical_multiplier


def build_real_disaster_world(cfg: dict) -> WorldState:
    world = make_scenario(
        seed=cfg["seed"],
        disaster_type="wildfire",
        n_shelters=cfg["n_shelters"],
        n_depots=cfg["n_depots"],
        n_transports=cfg["n_transports"],
        grid_size=cfg["grid_size"],
    )
    apply_population_surge(world, cfg["population_surge_factor"])
    return world


def run_optimizer_once_real(cfg: dict) -> float:
    world = build_real_disaster_world(cfg)
    rng = random.Random(cfg["seed"])
    total_unmet = 0.0
    outbreak_applied = False
    for t in range(cfg["timesteps"]):
        if not outbreak_applied and t >= cfg["disease_outbreak_timestep"]:
            apply_disease_outbreak(world, cfg["disease_outbreak_medical_multiplier"])
            outbreak_applied = True
        advance_disaster(world, rng, wildfire_spread_rate=cfg["wildfire_spread_rate"])
        alloc = solve_optimal_allocation(world)
        for (depot_id, shelter_id, resource), qty in alloc.items():
            world.depots[depot_id].stock[resource] = max(
                0.0, world.depots[depot_id].stock.get(resource, 0.0) - qty)
        total_unmet += unmet_urgent_need(world, alloc)
    return total_unmet


def _max_workers_for(model: str, n_items: int) -> int:
    """
    Local models (Ollama) almost always serve one request at a time
    regardless of how many you fire concurrently -- parallelizing those
    calls doesn't speed anything up and risks queuing/timeouts against a
    single local server. Hosted Claude calls are independent network
    round-trips and genuinely benefit from concurrency. Capped at 8 to
    avoid hammering the API with a burst far larger than any real client
    would send.
    """
    if model == "local":
        return 1
    return max(1, min(n_items, 8))


def _parse_reports_parallel(field_report_agent, world, rng, t, max_workers):
    # Situation-report TEXT GENERATION uses a single shared `rng` and must
    # stay single-threaded (Random isn't thread-safe and results must stay
    # reproducible given a seed). Only the PARSING step -- independent LLM
    # calls with no shared state -- runs concurrently.
    reports = {s.id: s.situation_report(rng) for s in world.shelters.values()}
    if max_workers == 1:
        return {sid: field_report_agent.parse(text, t) for sid, text in reports.items()}

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(field_report_agent.parse, text, t): sid
                   for sid, text in reports.items()}
        for fut in concurrent.futures.as_completed(futures):
            results[futures[fut]] = fut.result()

    # FIX: as_completed() yields in THREAD COMPLETION order, not a fixed
    # shelter order -- before Phase A this didn't matter (the Dispatcher's
    # output was discarded), but now that dict order can flow all the way
    # through to which deliveries actually get shipped when the vehicle
    # fleet can't cover every allocation, this was a real source of
    # run-to-run non-determinism unrelated to model sampling variance
    # (confirmed: parallel vs serial mode gave different unmet_need with
    # an otherwise-identical stub and seed). Rebuilding in `reports`'
    # canonical order (== world.shelters iteration order) makes downstream
    # behavior independent of which thread happened to finish first.
    return {sid: results[sid] for sid in reports}


def _allocate_parallel(depot_agent, world, shelter_urgency_estimates, t, max_workers, urgency_history):
    """
    Parallelizes DepotAgent.allocate() calls across depots -- safe to run
    concurrently since each call only reads (not mutates) shared state
    (depot.stock is depot-specific, transport_capacity here is just an
    aggregate estimate for the negotiator, not a hard reservation).

    NOTE: this only decides desired allocations. Actual shipping (the
    DispatcherAgent call that assigns real vehicles and mutates their
    busy_until) happens afterward, sequentially, in run_one_trial_real --
    that step is NOT safe to parallelize across depots, since depots share
    the same vehicle pool and concurrent dispatch calls could double-book
    a vehicle before either commits its busy_until.
    """
    def alloc_for_depot(depot):
        transport_capacity = sum(
            tr.capacity for tr in world.transports.values()
            if tr.busy_until <= t and not tr.destroyed
        )
        route_status = {
            shelter.id: not world.route_damaged(depot.position, shelter.position)
            for shelter in world.shelters.values()
        }
        route_risk = compute_route_risk_for_depot(world, depot)
        lookahead_context = {
            "timesteps_until_next_resupply": timesteps_until_next_resupply(world, depot),
            "expected_resupply_fraction": depot.resupply_fraction,
            "recent_avg_urgency_trend": list(urgency_history),
        }
        alloc = depot_agent.allocate(
            depot.id, depot.stock, transport_capacity, shelter_urgency_estimates, t,
            route_status=route_status, lookahead_context=lookahead_context,
            route_risk=route_risk)
        return depot.id, (alloc, route_risk)

    depots = list(world.depots.values())
    if max_workers == 1:
        return dict(alloc_for_depot(d) for d in depots)

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(alloc_for_depot, d) for d in depots]
        for fut in concurrent.futures.as_completed(futures):
            depot_id, result = fut.result()
            results[depot_id] = result
    return results


def run_one_trial_real(config_name: str, cfg: dict, api_key: str, optimal_unmet: float,
                        parallel: bool = True, verbose: bool = False):
    world = build_real_disaster_world(cfg)
    rng = random.Random(cfg["seed"])
    # RNG STREAM SEPARATION -- see FINDINGS.md "RNG Stream Contamination".
    risk_rng = random.Random(rng.random())
    fr_model, depot_model, dispatch_model = MODEL_CONFIGS[config_name]
    tracker = CostTracker()

    field_report_agent = FieldReportAgent(LLMBackend(fr_model, tracker, "field_report", api_key))
    depot_agent = DepotAgent(LLMBackend(depot_model, tracker, "depot_negotiation", api_key))
    dispatcher_agent = DispatcherAgent(LLMBackend(dispatch_model, tracker, "dispatcher", api_key))
    inter_depot_agent = InterDepotCoordinatorAgent(
        LLMBackend(depot_model, tracker, "inter_depot_coordination", api_key))

    fr_workers = _max_workers_for(fr_model, cfg["n_shelters"]) if parallel else 1
    depot_workers = _max_workers_for(depot_model, cfg["n_depots"]) if parallel else 1

    total_unmet = 0.0
    outbreak_applied = False
    urgency_history = []
    t0 = time.time()

    for t in range(cfg["timesteps"]):
        if not outbreak_applied and t >= cfg["disease_outbreak_timestep"]:
            apply_disease_outbreak(world, cfg["disease_outbreak_medical_multiplier"])
            outbreak_applied = True

        advance_disaster(world, rng, wildfire_spread_rate=cfg["wildfire_spread_rate"])

        arrived_this_step = resolve_arrivals(world, t)
        resolve_transfer_arrivals(world, t)

        shelter_urgency_estimates = _parse_reports_parallel(
            field_report_agent, world, rng, t, fr_workers)

        # Phase C: single coordination decision per timestep (not per depot,
        # not parallelized -- see agents/inter_depot_agent.py), run BEFORE
        # the parallel depot-allocation step below so a transfer's vehicle
        # usage is reflected in the transport availability that step sees.
        available_vehicles_for_coord = {
            tr.id: {"capacity": tr.capacity, "speed": tr.speed}
            for tr in world.transports.values() if tr.busy_until <= t and not tr.destroyed
        }
        depot_summaries = build_depot_summaries(world, shelter_urgency_estimates)
        transfer = inter_depot_agent.coordinate(depot_summaries, available_vehicles_for_coord, t)
        if transfer is not None:
            source = world.depots[transfer["source_depot_id"]]
            transfer_risk = route_risk_score(world, source.position,
                                              world.depots[transfer["dest_depot_id"]].position)
            apply_transfer_with_risk(world, risk_rng, source, transfer["dest_depot_id"],
                                      transfer["resource"], transfer["quantity"],
                                      transfer["vehicle_id"], t, transfer_risk)

        depot_allocations = _allocate_parallel(
            depot_agent, world, shelter_urgency_estimates, t, depot_workers, urgency_history)

        # Dispatch is sequential (see _allocate_parallel's docstring): each
        # depot's Dispatcher call re-reads current vehicle availability, so
        # processing depot-by-depot means the second depot correctly sees
        # vehicles the first depot just claimed this same round. Processing
        # ORDER rotates by timestep (see rotate_depots_for_fairness) so no
        # single depot always goes first and monopolizes the shared fleet --
        # confirmed bug, see FINDINGS.md "Depot Processing Order
        # Fleet-Starvation Bug".
        for depot in rotate_depots_for_fairness(world, t):
            desired_alloc, route_risk = depot_allocations.get(depot.id, ({}, {}))
            vehicle_assignments = dispatcher_agent.sequence(
                world, depot.id, desired_alloc, shelter_urgency_estimates, t,
                route_risk=route_risk)
            for vehicle_id, deliveries in vehicle_assignments.items():
                for delivery in deliveries:
                    shelter_id = delivery["shelter_id"]
                    resource = delivery["resource"]
                    qty = delivery["quantity"]
                    if qty <= 0:
                        continue
                    apply_shipment_with_risk(world, risk_rng, depot, shelter_id, resource, qty,
                                              vehicle_id, t, route_risk.get(shelter_id, 0.0))

        step_unmet = unmet_urgent_need(world, arrived_this_step)
        total_unmet += step_unmet

        urgency_history.append(round(average_urgency(shelter_urgency_estimates), 3))
        if len(urgency_history) > URGENCY_HISTORY_LENGTH:
            urgency_history.pop(0)

        if verbose:
            elapsed = time.time() - t0
            print(f"    t={t+1:>2}/{cfg['timesteps']}  "
                  f"cumulative_unmet={total_unmet:>12.1f}  elapsed={elapsed:>6.1f}s",
                  flush=True)

    wall_time = time.time() - t0
    gap_pct = ((total_unmet - optimal_unmet) / optimal_unmet * 100
               if optimal_unmet > 0 else float("nan"))
    return {
        "unmet_need": total_unmet,
        "optimality_gap_pct": gap_pct,
        "total_cost_usd": tracker.total_cost(),
        "wall_time_s": wall_time,
    }


def mean_std(values):
    values = [v for v in values if v is not None and v == v]
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS,
                         choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--repeats", type=int, default=3,
                         help="Repeats per config to quantify model-sampling noise "
                              "(same role as --repeats in run_sweep.py). Recommended "
                              ">=3 before trusting the comparison.")
    parser.add_argument("--wildfire-spread-rate", type=float,
                         default=LA_WILDFIRES_JAN2025["wildfire_spread_rate"])
    parser.add_argument("--population-surge-factor", type=float,
                         default=LA_WILDFIRES_JAN2025["population_surge_factor"])
    parser.add_argument("--timesteps", type=int, default=LA_WILDFIRES_JAN2025["timesteps"])
    parser.add_argument("--out", default="real_disaster_results.csv")
    parser.add_argument("--no-parallel", action="store_true",
                         help="Disable concurrent LLM calls (field-report parsing across "
                              "shelters, depot allocation across depots). Useful for "
                              "debugging or if you suspect concurrency itself is causing "
                              "issues (e.g. a local server that can't handle concurrent "
                              "requests well even at max_workers=1 fan-out).")
    parser.add_argument("--verbose", action="store_true",
                         help="Print cumulative unmet need and elapsed time after every "
                              "timestep, not just after each full run -- useful for seeing "
                              "that a long run is actually progressing.")
    args = parser.parse_args()

    cfg = dict(LA_WILDFIRES_JAN2025)
    cfg["wildfire_spread_rate"] = args.wildfire_spread_rate
    cfg["population_surge_factor"] = args.population_surge_factor
    cfg["timesteps"] = args.timesteps

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    needs_api = any(m != "local" for c in args.configs for m in MODEL_CONFIGS[c])
    if needs_api and not api_key:
        print("WARNING: ANTHROPIC_API_KEY not set. Configs using Haiku/Sonnet will fail. "
              "Run `export ANTHROPIC_API_KEY=sk-...` first, or restrict --configs to "
              "local_everywhere only.")

    print(f"Scenario: {cfg['description']}")
    print(f"  seed={cfg['seed']}  shelters={cfg['n_shelters']}  depots={cfg['n_depots']}  "
          f"transports={cfg['n_transports']}  grid={cfg['grid_size']}  timesteps={cfg['timesteps']}")
    print(f"  wildfire_spread_rate={cfg['wildfire_spread_rate']} (default 0.6)  "
          f"population_surge_factor={cfg['population_surge_factor']}x  "
          f"disease_outbreak at t={cfg['disease_outbreak_timestep']} "
          f"(medical x{cfg['disease_outbreak_medical_multiplier']})")
    print(f"  concurrency: {'disabled (--no-parallel)' if args.no_parallel else 'enabled (hosted-model calls only; local calls stay serial)'}")

    preview = build_real_disaster_world(cfg)
    total_pop = sum(s.population for s in preview.shelters.values())
    total_stock = sum(sum(d.stock.values()) for d in preview.depots.values())
    total_demand_per_t = sum(sum(s.consumption_rate.values()) for s in preview.shelters.values())
    stock_timesteps = total_stock / total_demand_per_t if total_demand_per_t > 0 else float("inf")
    print(f"  --> total simulated evacuee population: {total_pop}   "
          f"total starting depot stock (all resources): {total_stock:.0f}")
    print(f"  --> starting stock covers ~{stock_timesteps:.2f} timesteps of total demand "
          f"(scenario runs {cfg['timesteps']} timesteps)")
    if stock_timesteps < 1.5:
        print(f"  WARNING: stock/demand ratio is very low ({stock_timesteps:.2f}). Every config, "
              f"including the optimizer, may hit near-total resource exhaustion almost "
              f"immediately -- this reproduces the 'unrecoverable stockout' pathology already "
              f"diagnosed in FINDINGS.md issue #1, where no allocation strategy can differ once "
              f"nobody has anything left. Consider lowering --population-surge-factor before "
              f"spending time/API cost on a full run.\n")
    else:
        print()

    optimal_unmet = run_optimizer_once_real(cfg)
    print(f"Optimizer baseline (full-information LP): unmet_need={optimal_unmet:.2f}\n")

    raw_rows = []
    summary_rows = []

    for config_name in args.configs:
        print(f"=== {config_name} ===")
        trials = []
        for rep in range(args.repeats):
            try:
                result = run_one_trial_real(config_name, cfg, api_key, optimal_unmet,
                                             parallel=not args.no_parallel, verbose=args.verbose)
            except Exception as e:
                print(f"  repeat {rep+1}/{args.repeats} FAILED: {e}")
                raw_rows.append({"config": config_name, "repeat": rep, "status": "FAILED", "error": str(e)})
                continue
            trials.append(result)
            raw_rows.append({"config": config_name, "repeat": rep, "status": "OK", "error": "",
                              **{k: round(v, 6) for k, v in result.items()}})
            print(f"  repeat {rep+1}/{args.repeats}: unmet_need={result['unmet_need']:.2f}  "
                  f"gap={result['optimality_gap_pct']:+.1f}%  cost=${result['total_cost_usd']:.6f}")

        if not trials:
            summary_rows.append({"config": config_name, "status": "ALL_FAILED"})
            continue

        gap_mean, gap_std = mean_std([t["optimality_gap_pct"] for t in trials])
        unmet_mean, unmet_std = mean_std([t["unmet_need"] for t in trials])
        cost_mean, cost_std = mean_std([t["total_cost_usd"] for t in trials])
        toy_ref = TOY_SCALE_WILDFIRE_REFERENCE.get(config_name)
        summary_rows.append({
            "config": config_name,
            "n_ok": len(trials),
            "unmet_need_mean": round(unmet_mean, 2),
            "unmet_need_std": round(unmet_std, 2),
            "optimality_gap_pct_mean": round(gap_mean, 1),
            "optimality_gap_pct_std": round(gap_std, 1),
            "total_cost_usd_mean": round(cost_mean, 6),
            "toy_scale_wildfire_gap_pct_reference": toy_ref,
        })
        delta = (gap_mean - toy_ref) if toy_ref is not None else None
        if toy_ref is not None:
            print(f"  SUMMARY: gap={gap_mean:+.1f}% (std={gap_std:.1f})  "
                  f"cost=${cost_mean:.6f}  [toy-scale wildfire was {toy_ref:+.1f}%, "
                  f"delta={delta:+.1f}pp]\n")
        else:
            print(f"  SUMMARY: gap={gap_mean:+.1f}% (std={gap_std:.1f})  "
                  f"cost=${cost_mean:.6f}  [no toy-scale wildfire reference yet for this config]\n")

    if raw_rows:
        fields = sorted(set().union(*[r.keys() for r in raw_rows]))
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(raw_rows)

    print("=" * 100)
    print("REAL-DISASTER STRESS TEST SUMMARY  (vs. toy-scale wildfire:7 from FINDINGS.md)")
    print("=" * 100)
    header = (f"{'config':<38} {'gap% (mean±std)':>20} {'cost($)':>12} "
              f"{'toy-scale gap%':>16} {'delta(pp)':>10}")
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        if "optimality_gap_pct_mean" in row:
            toy_ref = row["toy_scale_wildfire_gap_pct_reference"]
            gap_str = f"{row['optimality_gap_pct_mean']:>+7.1f}% ± {row['optimality_gap_pct_std']:<8.1f}"
            if toy_ref is not None:
                delta = row["optimality_gap_pct_mean"] - toy_ref
                print(f"{row['config']:<38} {gap_str} "
                      f"{row['total_cost_usd_mean']:>10.6f}   "
                      f"{toy_ref:>+13.1f}%   {delta:>+8.1f}")
            else:
                print(f"{row['config']:<38} {gap_str} "
                      f"{row['total_cost_usd_mean']:>10.6f}   "
                      f"{'n/a':>14}   {'n/a':>8}")
        else:
            print(f"{row['config']:<38} {'ALL FAILED':>20}")
    print()
    print("How to read this:")
    print("  - Compare RANKING between configs here vs. the toy-scale sweep, not just")
    print("    absolute numbers -- this scenario is deliberately harsher (see module")
    print("    docstring), so worse absolute gaps are expected and not itself a failure.")
    print("  - If local_parsing_haiku_negotiation still tracks haiku_everywhere closely")
    print("    here (as it did at toy scale), that's real evidence the cost-efficiency")
    print("    finding generalizes to realistic scale/speed/scarcity, not just the")
    print("    original small synthetic setup.")
    print("  - If the ranking flips or the gap between them widens a lot, that's an")
    print("    important negative result: the finding may be scale-dependent, which")
    print("    is worth reporting honestly rather than papering over.")
    print(f"\nRaw per-run data written to {args.out}")


if __name__ == "__main__":
    main()