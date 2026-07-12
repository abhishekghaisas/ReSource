"""
A scenario calibrated specifically to the January 2025 PALISADES FIRE (not
the combined Palisades+Eaton framing used in run_real_disaster.py) using
real, named shelter and depot locations and their actual relative
distances, wherever that data exists -- and clearly labeling every place a
judgment call had to fill a gap. See project conversation history for the
full research trail; summary below.

WHAT'S REAL VS. ESTIMATED
-------------------------------------------------------------------------
REAL (sourced from CAL FIRE incident updates, LAFD, City of Malibu, LA
County Recovery -- a separate fabricated-looking source claiming named
"Pacific Palisades Community Shelter" locations with 555-prefix phone
numbers was identified and discarded):
  - 3 shelters: Westwood Recreation Center, El Camino Real Charter High
    School, Pasadena Convention Center -- the only three confirmed
    overnight shelter locations for Palisades Fire evacuees. (A fourth
    candidate, Cheviot Hills Recreation Center, was excluded: its
    documented role was missing-persons/family reunification, not
    sheltering or supply distribution -- a different function, not
    padding for a bigger number.)
  - 2 depots: UCLA Research Park West (the real Westside Disaster Resource
    Center) and the Malibu Pier staging area (the real resident/supply
    checkpoint for the Malibu portion of the burn zone).
  - Grid positions: computed from each location's approximate real-world
    coordinates via an equirectangular projection (2 miles/grid-unit),
    preserving actual relative distances -- e.g. Westwood and UCLA are
    genuinely ~1.3 real miles apart and land 1 grid-unit apart; Malibu
    Pier to Pasadena Convention Center is genuinely the longest span
    (~31 real miles) and is the longest distance on the grid too.
    Coordinates are approximated from general geography, not a geocoding
    lookup -- flagged as approximate, not exact.
  - Fire ignition point (epicenter): approximate real location in the
    Santa Monica Mountains near Pacific Palisades.
  - Timesteps: 24, matching the real 24-day containment window (Jan 7-31).

ESTIMATED (no sourced figure found; each is a clearly-labeled judgment
call, not a reported fact):
  - Fleet size (5 vehicles): no source reports a relief-supply vehicle
    count for this event (as opposed to firefighting apparatus, a
    different fleet entirely). Kept at the same order of magnitude used
    throughout this project's other scenarios.
  - Shelter population (150 each, 450 total): real reporting gives a
    combined PEAK overnight occupancy of ~450 across shelters, with no
    per-shelter breakdown -- split evenly across the 3 confirmed shelters
    as the simplest defensible assumption, not because occupancy was
    actually even.
  - Depot stock: left at simulation.world.make_scenario()'s natural
    random default (150-300/resource/depot) rather than an artificial
    scarcity multiplier. This is a deliberate choice: with population now
    realistically small (450, not an inflated synthetic surge), it's
    already close in scale to the original validated toy-scale sweep, so
    the artificial "population_surge_factor" scarcity engineering used in
    the earlier (combined, synthetic-scale) real-disaster scenario isn't
    needed here -- whatever difficulty exists should come from the fire's
    own dynamics, not a manufactured resource shortfall.
  - Fire radius calibration (RECALIBRATED -- wildfire_spread_rate=0.0711
    grid-units/timestep, was 0.5): the original rate was chosen ONLY to
    keep Pasadena's distance from the epicenter unreached through day 24
    -- nobody checked whether the circle's ABSOLUTE size was realistic at
    any point along the way. It wasn't: by day 24 by the old rate, the
    "fire radius" reached 12 grid-units (24 miles), a circle of ~1,800 sq
    mi -- roughly 50x the real fire's actual footprint, visibly covering
    the Pacific Ocean and reaching into the San Fernando Valley once
    plotted on a real map (a real map is what caught this; the earlier
    abstract grid view hid it completely).

    Recalibrated from the real fire's actual documented footprint --
    23,448 acres (36.6 sq mi) -- treated as an equivalent circle:
    radius = sqrt(36.6 / pi) ~= 3.41 mi ~= 1.71 grid-units. Reaching that
    by day 24 (full containment) gives wildfire_spread_rate =
    1.71/24 ~= 0.0711 grid-units/timestep. This keeps the same constant-
    rate simplification documented above (the real fire's growth was far
    more front-loaded than any constant rate captures), but now at least
    the circle's SIZE is realistic throughout, not just its relationship
    to Pasadena specifically.

    A natural worry going into this fix: would a realistically-sized fire
    ever come close enough to any real route to matter, given every
    location here sits 8-26 miles from the epicenter? Checked directly
    (see FINDINGS.md "Fire Radius Realism Fix") -- yes, meaningfully so.
    Depots and shelters sit in different directions AROUND the epicenter,
    not clustered on one far side of it, so several straight-line paths
    between them still pass close to the center even though no single
    endpoint does. Of the 7 real depot-shelter/depot-depot routes: 3
    eventually become hard-blocked (day 15-21, notably later than the old
    rate's day 3-11), 3 stay risky-but-passable the entire 24 days, and
    exactly one (D1-S3, UCLA-Pasadena) is never affected at all --
    correctly matching the real-world fact that Pasadena was Eaton Fire
    territory, not Palisades. Every shelter keeps at least one viable
    (if sometimes risky) route to at least one depot for the full 24
    days -- total shelter isolation, which the old miscalibration
    produced by day 11, does not happen once the fire is sized correctly.

Usage:
    python run_palisades_scenario.py --configs local_everywhere --repeats 1 --verbose
    python run_palisades_scenario.py --repeats 3
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
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

# Real-location grid positions, computed via equirectangular projection at
# 2 miles/grid-unit -- see module docstring. Shelter/depot IDs (S1, S2, ...,
# D1, D2) match simulation.world.make_scenario()'s creation order (i=0,1,2
# for shelters, i=0,1 for depots), so overriding by key is unambiguous.
MILES_PER_GRID_UNIT = 2.0  # same projection scale used to derive REAL_POSITIONS below
REAL_POSITIONS = {
    "S1": (9, 3),   # Westwood Recreation Center
    "S2": (4, 7),   # El Camino Real Charter High School
    "S3": (17, 6),  # Pasadena Convention Center
    "D1": (9, 2),   # UCLA Research Park West (Disaster Resource Center)
    "D2": (2, 2),   # Malibu Pier staging area
}
REAL_LOCATION_NAMES = {
    "S1": "Westwood Recreation Center",
    "S2": "El Camino Real Charter High School",
    "S3": "Pasadena Convention Center",
    "D1": "UCLA Research Park West",
    "D2": "Malibu Pier staging area",
}
EPICENTER = (6, 4)  # approximate real ignition point, Santa Monica Mountains

PALISADES_FIRE_JAN2025 = {
    "description": "Calibrated specifically to the Palisades Fire (not combined with Eaton)",
    "seed": 2025,
    "n_shelters": 3,
    "n_depots": 2,
    "n_transports": 5,             # ESTIMATE -- no sourced fleet count, see module docstring
    "grid_size": (19, 9),
    "timesteps": 24,                # REAL -- 24-day containment window (Jan 7-31)
    "wildfire_spread_rate": 0.0711,  # RECALIBRATED -- see module docstring, "Fire radius calibration"
    "shelter_population": 150.0,    # ESTIMATE -- even split of ~450 combined peak occupancy
}


def load_real_routes(world: WorldState, cache_path: str) -> dict:
    """
    Loads real road-network distances from fetch_real_routes.py's cache
    (see that script's docstring for why OSRM, and why this is a one-time
    fetch rather than a live call). Converts each route's real driving
    distance (meters) into grid-unit equivalents using the SAME 2 mi/
    grid-unit projection scale REAL_POSITIONS was derived from, then
    populates world.real_road_distances so dispatch_shipment/
    dispatch_transfer use real travel time instead of straight-line
    distance -- see world.py's effective_distance() docstring for why
    fire risk/blocking deliberately does NOT use this (a fire doesn't
    spread along roads).

    Returns the route geometry dict ({pair_key: [[lat,lon],...]}) for
    callers that want to include it in replay output for map display --
    this function's caller decides what to do with it, it isn't stored on
    world itself since geometry is a display concern, not a simulation one.

    Missing cache file or missing individual pairs fail gracefully (a
    warning is printed, that specific pair falls back to straight-line
    distance) rather than crashing the whole run.
    """
    geometries = {}
    try:
        with open(cache_path) as f:
            cache = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"WARNING: could not load real-routes cache from {cache_path} ({e}) -- "
              f"falling back to straight-line distance for every route.")
        return geometries

    for pair_key, route in cache.get("routes", {}).items():
        if route is None:
            continue  # this specific pair failed to fetch; already warned at fetch time
        a_id, b_id = pair_key.split("-")
        if a_id not in REAL_POSITIONS or b_id not in REAL_POSITIONS:
            continue
        distance_miles = route["distance_m"] / 1609.34
        distance_grid_units = distance_miles / MILES_PER_GRID_UNIT
        world.real_road_distances[(REAL_POSITIONS[a_id], REAL_POSITIONS[b_id])] = distance_grid_units
        geometries[pair_key] = route["geometry_latlon"]
    return geometries


def build_palisades_world(cfg: dict) -> WorldState:
    world = make_scenario(
        seed=cfg["seed"], disaster_type="wildfire",
        n_shelters=cfg["n_shelters"], n_depots=cfg["n_depots"],
        n_transports=cfg["n_transports"], grid_size=cfg["grid_size"],
    )
    for sid, pos in REAL_POSITIONS.items():
        if sid in world.shelters:
            world.shelters[sid].position = pos
        elif sid in world.depots:
            world.depots[sid].position = pos
    world.epicenter = EPICENTER

    # Set shelter population to the real (even-split) estimate, rescaling
    # consumption_rate proportionally -- same reasoning as
    # run_real_disaster.py's apply_population_surge: per-capita consumption
    # rate is what's actually meaningful, not the absolute population number
    # alone, so both must move together to stay internally consistent.
    for shelter in world.shelters.values():
        factor = cfg["shelter_population"] / shelter.population
        shelter.population = cfg["shelter_population"]
        for r in RESOURCE_TYPES:
            shelter.consumption_rate[r] = shelter.consumption_rate.get(r, 0.0) * factor

    return world


def run_optimizer_once(cfg: dict, real_routes_cache: str = None) -> float:
    world = build_palisades_world(cfg)
    if real_routes_cache:
        load_real_routes(world, real_routes_cache)
    rng = random.Random(cfg["seed"])
    total_unmet = 0.0
    for t in range(cfg["timesteps"]):
        advance_disaster(world, rng, wildfire_spread_rate=cfg["wildfire_spread_rate"])
        alloc = solve_optimal_allocation(world)
        for (depot_id, shelter_id, resource), qty in alloc.items():
            world.depots[depot_id].stock[resource] = max(
                0.0, world.depots[depot_id].stock.get(resource, 0.0) - qty)
        total_unmet += unmet_urgent_need(world, alloc)
    return total_unmet


def _max_workers_for(model: str, n_items: int) -> int:
    if model == "local":
        return 1
    return max(1, min(n_items, 8))


def _parse_reports_parallel(field_report_agent, world, rng, t, max_workers):
    reports = {s.id: s.situation_report(rng) for s in world.shelters.values()}
    if max_workers == 1:
        return {sid: field_report_agent.parse(text, t) for sid, text in reports.items()}
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(field_report_agent.parse, text, t): sid
                   for sid, text in reports.items()}
        for fut in concurrent.futures.as_completed(futures):
            results[futures[fut]] = fut.result()
    return {sid: results[sid] for sid in reports}  # deterministic order, see run_real_disaster.py


def run_one_trial(config_name: str, cfg: dict, api_key: str, optimal_unmet: float,
                   parallel: bool = True, verbose: bool = False,
                   disable_lookahead: bool = False, disable_interdepot: bool = False,
                   capture_replay: bool = False, real_routes_cache: str = None):
    world = build_palisades_world(cfg)
    route_geometries = load_real_routes(world, real_routes_cache) if real_routes_cache else {}
    rng = random.Random(cfg["seed"])
    # RNG STREAM SEPARATION -- see FINDINGS.md "RNG Stream Contamination".
    # This was the exact scenario that surfaced the bug: haiku_everywhere
    # attempts far more risky shipments than local_everywhere, and sharing
    # one rng stream for both world-ground-truth draws (population growth)
    # and risk-outcome draws meant the two configs were silently facing
    # DIFFERENT population trajectories despite the same seed -- confirmed
    # directly (S1 population diverged to 161 vs. 221 by day 20 before this
    # fix). Derived immediately, before any config-dependent draws happen.
    risk_rng = random.Random(rng.random())
    fr_model, depot_model, dispatch_model = MODEL_CONFIGS[config_name]
    tracker = CostTracker()

    field_report_agent = FieldReportAgent(LLMBackend(fr_model, tracker, "field_report", api_key))
    depot_agent = DepotAgent(LLMBackend(depot_model, tracker, "depot_negotiation", api_key))
    dispatcher_agent = DispatcherAgent(LLMBackend(dispatch_model, tracker, "dispatcher", api_key))
    inter_depot_agent = InterDepotCoordinatorAgent(
        LLMBackend(depot_model, tracker, "inter_depot_coordination", api_key))

    fr_workers = _max_workers_for(fr_model, cfg["n_shelters"]) if parallel else 1
    total_unmet = 0.0
    urgency_history = []
    transfers_made = []  # for diagnosis: did Phase C ever actually fire in this run?
    all_risk_outcomes = []  # Phase R: accumulated across the whole run, for diagnosis
    replay_frames = [] if capture_replay else None
    t0 = time.time()

    for t in range(cfg["timesteps"]):
        advance_disaster(world, rng, wildfire_spread_rate=cfg["wildfire_spread_rate"])
        arrived_this_step = resolve_arrivals(world, t)
        transfers_arrived_this_step = resolve_transfer_arrivals(world, t)

        shelter_urgency_estimates = _parse_reports_parallel(
            field_report_agent, world, rng, t, fr_workers)

        shipments_dispatched_this_step = []
        transfers_dispatched_this_step = []
        risk_outcomes_this_step = []  # Phase R: for replay/diagnostics visibility
        vehicles_destroyed_this_step = []

        # Phase C: one coordination decision per timestep, before per-depot
        # negotiation, so a transfer's vehicle usage is reflected in the
        # transport availability the per-depot loop sees this round. This
        # is the highest-value scenario for Phase C in the project so far --
        # two real, geographically distinct depots (UCLA Research Park West
        # and Malibu Pier), not a synthetic setup.
        if not disable_interdepot:
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
                outcome = apply_transfer_with_risk(
                    world, risk_rng, source, transfer["dest_depot_id"], transfer["resource"],
                    transfer["quantity"], transfer["vehicle_id"], t, transfer_risk)
                transfers_made.append((t, transfer))
                transfers_dispatched_this_step.append(dict(transfer))
                if transfer_risk > 0:
                    risk_outcomes_this_step.append({
                        "kind": "transfer", "vehicle_id": transfer["vehicle_id"],
                        "risk": round(transfer_risk, 3), "outcome": outcome,
                    })
                if outcome == "total_loss":
                    vehicles_destroyed_this_step.append(transfer["vehicle_id"])

        for depot in rotate_depots_for_fairness(world, t):
            transport_capacity = sum(
                tr.capacity for tr in world.transports.values()
                if tr.busy_until <= t and not tr.destroyed
            )
            route_status = {
                shelter.id: not world.route_damaged(depot.position, shelter.position)
                for shelter in world.shelters.values()
            }
            route_risk = compute_route_risk_for_depot(world, depot)
            lookahead_context = None if disable_lookahead else {
                "timesteps_until_next_resupply": timesteps_until_next_resupply(world, depot),
                "expected_resupply_fraction": depot.resupply_fraction,
                "recent_avg_urgency_trend": list(urgency_history),
            }
            desired_alloc = depot_agent.allocate(
                depot.id, depot.stock, transport_capacity, shelter_urgency_estimates, t,
                route_status=route_status,
                lookahead_context=lookahead_context, route_risk=route_risk)
            vehicle_assignments = dispatcher_agent.sequence(
                world, depot.id, desired_alloc, shelter_urgency_estimates, t,
                route_risk=route_risk)
            for vehicle_id, deliveries in vehicle_assignments.items():
                for delivery in deliveries:
                    qty = delivery["quantity"]
                    if qty <= 0:
                        continue
                    resource = delivery["resource"]
                    shelter_id = delivery["shelter_id"]
                    risk = route_risk.get(shelter_id, 0.0)
                    outcome = apply_shipment_with_risk(world, risk_rng, depot, shelter_id, resource,
                                                        qty, vehicle_id, t, risk)
                    if capture_replay:
                        shipments_dispatched_this_step.append({
                            "depot_id": depot.id, "shelter_id": shelter_id,
                            "resource": resource, "quantity": round(qty, 2),
                            "risk": round(risk, 3), "outcome": outcome,
                        })
                    if risk > 0:
                        risk_outcomes_this_step.append({
                            "kind": "shipment", "vehicle_id": vehicle_id,
                            "risk": round(risk, 3), "outcome": outcome,
                        })
                    if outcome == "total_loss":
                        vehicles_destroyed_this_step.append(vehicle_id)

        all_risk_outcomes.extend(risk_outcomes_this_step)

        step_unmet = unmet_urgent_need(world, arrived_this_step)
        total_unmet += step_unmet

        if capture_replay:
            replay_frames.append({
                "t": t,
                "fire_radius": round(cfg["wildfire_spread_rate"] * (t + 1), 2),
                "blocked_route_pairs": [
                    [did, sid] for did in world.depots for sid in world.shelters
                    if world.route_damaged(world.depots[did].position, world.shelters[sid].position)
                ],
                "shelters": {
                    sid: {
                        "population": s.population,
                        "ground_truth_urgency": {r: round(s.urgency(r), 3) for r in RESOURCE_TYPES},
                        "reported_urgency": {k: v for k, v in
                                              shelter_urgency_estimates.get(sid, {}).items()
                                              if k in RESOURCE_TYPES},
                        # Shelter's own on-hand stock -- urgency() is computed
                        # directly from this, so logging it lets the exact
                        # unmet_urgent_need contribution be reconstructed and
                        # verified per shelter/resource/day, not just inferred.
                        "own_stock": {r: round(s.stock.get(r, 0.0), 2) for r in RESOURCE_TYPES},
                        "consumption_rate": {r: round(s.consumption_rate.get(r, 0.0), 3) for r in RESOURCE_TYPES},
                    } for sid, s in world.shelters.items()
                },
                "depots": {did: {"stock": {r: round(v, 1) for r, v in d.stock.items()}}
                           for did, d in world.depots.items()},
                "shipments_dispatched": shipments_dispatched_this_step,
                "transfers_dispatched": transfers_dispatched_this_step,
                # NEW: what actually ARRIVED this timestep (what
                # unmet_urgent_need is actually evaluated against) --
                # distinct from shipments_dispatched, which only records
                # when something LEFT a depot. A shipment dispatched on day
                # N doesn't necessarily arrive on day N; this closes that
                # gap for anyone trying to trace the metric exactly.
                "arrivals_this_step": [
                    {"depot_id": did, "shelter_id": sid, "resource": r, "quantity": round(qty, 2)}
                    for (did, sid, r), qty in arrived_this_step.items()
                ],
                "risk_outcomes": risk_outcomes_this_step,
                "vehicles_destroyed": vehicles_destroyed_this_step,
                "cumulative_unmet": round(total_unmet, 1),
                "step_unmet": round(step_unmet, 1),
            })

        urgency_history.append(round(average_urgency(shelter_urgency_estimates), 3))
        if len(urgency_history) > URGENCY_HISTORY_LENGTH:
            urgency_history.pop(0)

        if verbose:
            elapsed = time.time() - t0
            print(f"    t={t+1:>2}/{cfg['timesteps']}  cumulative_unmet={total_unmet:>10.1f}  "
                  f"elapsed={elapsed:>6.1f}s", flush=True)

    wall_time = time.time() - t0
    gap_pct = ((total_unmet - optimal_unmet) / optimal_unmet * 100
               if optimal_unmet > 0 else float("nan"))
    n_total_loss = sum(1 for r in all_risk_outcomes if r["outcome"] == "total_loss")
    n_partial_loss = sum(1 for r in all_risk_outcomes if r["outcome"] == "partial_loss")
    return {
        "unmet_need": total_unmet,
        "optimality_gap_pct": gap_pct,
        "total_cost_usd": tracker.total_cost(),
        "wall_time_s": wall_time,
        "n_transfers": len(transfers_made),
        "transfers": transfers_made,
        "n_risky_attempts": len(all_risk_outcomes),
        "n_vehicles_lost": n_total_loss,
        "n_partial_losses": n_partial_loss,
        "replay_frames": replay_frames,
        "route_geometries": route_geometries,
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
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timesteps", type=int, default=PALISADES_FIRE_JAN2025["timesteps"])
    parser.add_argument("--out", default="palisades_scenario_results.csv")
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--disable-lookahead", action="store_true",
                         help="Ablation: run without Phase D lookahead_context (DepotAgent "
                              "gets no resupply-timing/urgency-trend info), to isolate its effect.")
    parser.add_argument("--disable-interdepot", action="store_true",
                         help="Ablation: run without Phase C inter-depot coordination (no "
                              "transfers ever considered), to isolate its effect.")
    parser.add_argument("--replay-log", default=None,
                         help="Path to write a per-timestep replay trace (JSON) for dashboard "
                              "visualization -- only captured on the FIRST repeat of each "
                              "config (representative run, not statistical repeats).")
    parser.add_argument("--real-routes-cache", default=None,
                         help="Path to real_routes_cache.json (from fetch_real_routes.py). "
                              "When provided, travel time uses real road-network distance "
                              "instead of straight-line grid distance for the 7 relevant "
                              "location pairs. Fire risk/blocking is unaffected -- it "
                              "intentionally keeps using straight-line proximity to the "
                              "epicenter regardless of this flag.")
    args = parser.parse_args()

    cfg = dict(PALISADES_FIRE_JAN2025)
    cfg["timesteps"] = args.timesteps

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    needs_api = any(m != "local" for c in args.configs for m in MODEL_CONFIGS[c])
    if needs_api and not api_key:
        print("WARNING: ANTHROPIC_API_KEY not set. Configs using Haiku will fail.")

    print(f"Scenario: {cfg['description']}")
    print(f"  Real locations:")
    for key, name in REAL_LOCATION_NAMES.items():
        role = "shelter" if key.startswith("S") else "depot"
        print(f"    {key} ({role}): {name} @ grid{REAL_POSITIONS[key]}")
    print(f"  Fire epicenter (approx. real ignition point): grid{EPICENTER}")
    print(f"  seed={cfg['seed']}  grid={cfg['grid_size']}  timesteps={cfg['timesteps']}  "
          f"transports={cfg['n_transports']} (ESTIMATE)")
    print(f"  wildfire_spread_rate={cfg['wildfire_spread_rate']}  "
          f"shelter_population={cfg['shelter_population']:.0f} each (ESTIMATE, even split)")

    preview = build_palisades_world(cfg)
    total_pop = sum(s.population for s in preview.shelters.values())
    total_stock = sum(sum(d.stock.values()) for d in preview.depots.values())
    total_demand_per_t = sum(sum(s.consumption_rate.values()) for s in preview.shelters.values())
    print(f"  --> total shelter population: {total_pop:.0f}   "
          f"total starting depot stock: {total_stock:.0f}   "
          f"stock covers ~{total_stock/total_demand_per_t:.2f} timesteps of demand\n")

    optimal_unmet = run_optimizer_once(cfg, real_routes_cache=args.real_routes_cache)
    print(f"Optimizer baseline (full-information LP): unmet_need={optimal_unmet:.2f}\n")

    raw_rows = []
    summary_rows = []
    replay_log = {} if args.replay_log else None
    route_geometries_captured = {}

    for config_name in args.configs:
        print(f"=== {config_name} ===")
        trials = []
        for rep in range(args.repeats):
            try:
                result = run_one_trial(config_name, cfg, api_key, optimal_unmet,
                                        parallel=not args.no_parallel, verbose=args.verbose,
                                        disable_lookahead=args.disable_lookahead,
                                        disable_interdepot=args.disable_interdepot,
                                        capture_replay=(args.replay_log is not None and rep == 0),
                                        real_routes_cache=args.real_routes_cache)
            except Exception as e:
                print(f"  repeat {rep+1}/{args.repeats} FAILED: {e}")
                raw_rows.append({"config": config_name, "repeat": rep, "status": "FAILED", "error": str(e)})
                continue
            if args.replay_log is not None and rep == 0:
                replay_log[config_name] = result["replay_frames"]
                if result.get("route_geometries"):
                    route_geometries_captured = result["route_geometries"]
            trials.append(result)
            raw_rows.append({"config": config_name, "repeat": rep, "status": "OK", "error": "",
                              **{k: round(v, 6) for k, v in result.items()
                                 if k not in ("transfers", "replay_frames", "route_geometries")}})
            print(f"  repeat {rep+1}/{args.repeats}: unmet_need={result['unmet_need']:.2f}  "
                  f"gap={result['optimality_gap_pct']:+.1f}%  cost=${result['total_cost_usd']:.6f}  "
                  f"transfers={result['n_transfers']}  "
                  f"risky_attempts={result['n_risky_attempts']} "
                  f"(vehicles_lost={result['n_vehicles_lost']}, partial_losses={result['n_partial_losses']})")

        if not trials:
            summary_rows.append({"config": config_name, "status": "ALL_FAILED"})
            continue

        gap_mean, gap_std = mean_std([t["optimality_gap_pct"] for t in trials])
        cost_mean, _ = mean_std([t["total_cost_usd"] for t in trials])
        summary_rows.append({
            "config": config_name, "n_ok": len(trials),
            "optimality_gap_pct_mean": round(gap_mean, 1),
            "optimality_gap_pct_std": round(gap_std, 1),
            "total_cost_usd_mean": round(cost_mean, 6),
        })
        print(f"  SUMMARY: gap={gap_mean:+.1f}% (std={gap_std:.1f})  cost=${cost_mean:.6f}\n")

    if raw_rows:
        fields = sorted(set().union(*[r.keys() for r in raw_rows]))
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(raw_rows)

    print("=" * 90)
    print("PALISADES FIRE SCENARIO SUMMARY")
    print("=" * 90)
    header = f"{'config':<42} {'gap% (mean±std)':>20} {'cost($)':>12}"
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        if "optimality_gap_pct_mean" in row:
            print(f"{row['config']:<42} {row['optimality_gap_pct_mean']:>+7.1f}% ± "
                  f"{row['optimality_gap_pct_std']:<8.1f} {row['total_cost_usd_mean']:>10.6f}")
        else:
            print(f"{row['config']:<42} {'ALL FAILED':>20}")
    print(f"\nRaw per-run data written to {args.out}")

    if replay_log is not None:
        output = {
            "scenario": {
                "description": cfg["description"],
                "locations": {
                    key: {"name": REAL_LOCATION_NAMES[key], "position": list(pos),
                          "role": "shelter" if key.startswith("S") else "depot"}
                    for key, pos in REAL_POSITIONS.items()
                },
                "epicenter": list(EPICENTER),
                "grid_size": list(cfg["grid_size"]),
                "timesteps": cfg["timesteps"],
                # Real road-network geometry (lat/lon point lists), keyed by
                # "D1-S1" etc. -- empty dict if --real-routes-cache wasn't
                # provided, in which case a dashboard should fall back to
                # drawing straight lines between the fixed lat/lng points.
                "route_geometries": route_geometries_captured,
            },
            "summary_by_config": {
                row["config"]: {
                    "gap_pct_mean": row.get("optimality_gap_pct_mean"),
                    "gap_pct_std": row.get("optimality_gap_pct_std"),
                    "cost_usd_mean": row.get("total_cost_usd_mean"),
                }
                for row in summary_rows if "optimality_gap_pct_mean" in row
            },
            "frames_by_config": replay_log,
        }
        with open(args.replay_log, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Replay trace (for dashboard) written to {args.replay_log}")


if __name__ == "__main__":
    main()