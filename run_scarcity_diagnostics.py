"""
Tests the "supply-ceiling saturation" hypothesis for why local_everywhere,
haiku_everywhere, and local_parsing_haiku_negotiation converged to the exact
same +3.2% gap on the LA-wildfires-calibrated real-disaster scenario, even
AFTER fixing the DepotAgent clamp-order bug (see depot_agent.py and the
project conversation history -- that fix was real and necessary, but did not
change the convergence, which rules it out as the explanation).

HYPOTHESIS BEING TESTED
-------------------------------------------------------------------------
Starting depot stock covers only ~2.58 timesteps of total demand, and
resupply replaces only ~1.3 timesteps of demand per 5-timestep interval
(see run_real_disaster.py's module docstring). If nearly every shelter's
urgency crosses the threshold that counts toward unmet_urgent_need() by
mid-scenario, the aggregate metric becomes dominated by "how much total
resource was available to deliver" (a supply-side ceiling that's identical
regardless of which model is negotiating) rather than "which specific
shelter got prioritized" (the thing model quality could actually affect).
In other words: there may not be enough slack left in the scenario for
negotiation intelligence to matter, regardless of how good or bad it is.

WHAT THIS SCRIPT MEASURES, PER TIMESTEP, PER CONFIG
-------------------------------------------------------------------------
  1. total_delivered vs. total_available (depot stock immediately after
     that timestep's resupply, before allocation). If delivered stays
     pinned near ~100% of available for every config every timestep, that's
     the supply-ceiling signature: everyone is maxing out the same limited
     pool regardless of negotiation quality.
  2. pct_shelters_critical: fraction of shelters with urgency >= 0.5 (the
     same threshold unmet_urgent_need() uses) on at least one resource. If
     this saturates near 100% early and stays there, there's no meaningful
     "who deserves it more" distinction left for a negotiator to exploit --
     nearly everyone already qualifies as critical.

If both signatures show up consistently across local/Haiku/mixed, that's
strong evidence the convergence is a genuine property of the scenario's
scarcity level, not a bug or an artifact of one particular model being
secretly as good as another.

Usage:
    python run_scarcity_diagnostics.py
    python run_scarcity_diagnostics.py --configs haiku_everywhere local_parsing_haiku_negotiation
"""

from __future__ import annotations

import argparse
import csv
import os
import random

from agents.base_agent import CostTracker, LLMBackend
from agents.depot_agent import DepotAgent
from agents.dispatcher_agent import DispatcherAgent
from agents.field_report_agent import FieldReportAgent
from agents.inter_depot_agent import InterDepotCoordinatorAgent
from optimizer.baseline import solve_optimal_allocation, unmet_urgent_need
from run_real_disaster import (
    LA_WILDFIRES_JAN2025,
    apply_disease_outbreak,
    apply_population_surge,
    run_one_trial_real,
)
from run_simulation import (MODEL_CONFIGS, URGENCY_HISTORY_LENGTH, average_urgency,
                             build_depot_summaries, compute_route_risk_for_depot,
                             apply_shipment_with_risk, apply_transfer_with_risk,
                             rotate_depots_for_fairness)
from simulation.world import (RESOURCE_TYPES, advance_disaster, dispatch_shipment,
                               dispatch_transfer, make_scenario, resolve_arrivals,
                               resolve_transfer_arrivals, route_risk_score,
                               timesteps_until_next_resupply)

URGENCY_THRESHOLD = 0.5  # matches optimizer.baseline.unmet_urgent_need's default


def total_depot_stock(world) -> float:
    return sum(sum(depot.stock.values()) for depot in world.depots.values())


def pct_shelters_critical(world) -> float:
    n_critical = sum(
        1 for s in world.shelters.values()
        if max(s.urgency(r) for r in RESOURCE_TYPES) >= URGENCY_THRESHOLD
    )
    return 100.0 * n_critical / len(world.shelters)


def build_world(cfg: dict):
    world = make_scenario(
        seed=cfg["seed"], disaster_type="wildfire",
        n_shelters=cfg["n_shelters"], n_depots=cfg["n_depots"],
        n_transports=cfg["n_transports"], grid_size=cfg["grid_size"],
    )
    apply_population_surge(world, cfg["population_surge_factor"])
    return world


def run_optimizer_with_trace(cfg: dict) -> list[dict]:
    world = build_world(cfg)
    rng = random.Random(cfg["seed"])
    trace = []
    outbreak_applied = False
    for t in range(cfg["timesteps"]):
        if not outbreak_applied and t >= cfg["disease_outbreak_timestep"]:
            apply_disease_outbreak(world, cfg["disease_outbreak_medical_multiplier"])
            outbreak_applied = True
        advance_disaster(world, rng, wildfire_spread_rate=cfg["wildfire_spread_rate"])

        total_available = total_depot_stock(world)
        critical_pct = pct_shelters_critical(world)

        alloc = solve_optimal_allocation(world)
        total_delivered = sum(alloc.values())
        for (depot_id, shelter_id, resource), qty in alloc.items():
            world.depots[depot_id].stock[resource] = max(
                0.0, world.depots[depot_id].stock.get(resource, 0.0) - qty)

        trace.append({
            "config": "optimizer_baseline",
            "timestep": t,
            "total_available": round(total_available, 2),
            "total_delivered": round(total_delivered, 2),
            "delivered_pct_of_available": round(
                100.0 * total_delivered / total_available, 1) if total_available > 0 else 0.0,
            "pct_shelters_critical": round(critical_pct, 1),
            "unmet_need_this_step": round(unmet_urgent_need(world, alloc), 2),
        })
    return trace


def run_agent_config_with_trace(config_name: str, cfg: dict, api_key: str) -> list[dict]:
    fr_model, depot_model, dispatch_model = MODEL_CONFIGS[config_name]
    tracker = CostTracker()
    world = build_world(cfg)
    rng = random.Random(cfg["seed"])
    # RNG STREAM SEPARATION -- see FINDINGS.md "RNG Stream Contamination".
    risk_rng = random.Random(rng.random())

    field_report_agent = FieldReportAgent(LLMBackend(fr_model, tracker, "field_report", api_key))
    depot_agent = DepotAgent(LLMBackend(depot_model, tracker, "depot_negotiation", api_key))
    dispatcher_agent = DispatcherAgent(LLMBackend(dispatch_model, tracker, "dispatcher", api_key))
    inter_depot_agent = InterDepotCoordinatorAgent(
        LLMBackend(depot_model, tracker, "inter_depot_coordination", api_key))

    trace = []
    outbreak_applied = False
    urgency_history = []
    for t in range(cfg["timesteps"]):
        if not outbreak_applied and t >= cfg["disease_outbreak_timestep"]:
            apply_disease_outbreak(world, cfg["disease_outbreak_medical_multiplier"])
            outbreak_applied = True
        advance_disaster(world, rng, wildfire_spread_rate=cfg["wildfire_spread_rate"])

        total_available = total_depot_stock(world)
        critical_pct = pct_shelters_critical(world)

        # Phase A: "delivered" now has two different meanings that used to
        # be the same instant. `arrived_this_step` (what actually reaches a
        # shelter, possibly dispatched several timesteps ago) is what
        # unmet_urgent_need() should be scored against. `shipped_this_step`
        # (what leaves depot.stock THIS round via a real vehicle) is what
        # this diagnostic's "delivered/available" utilization comparison
        # actually wants -- it was always asking "how much of the currently
        # sitting stock gets put to use this round," not "how much reaches
        # a shelter this exact instant."
        arrived_this_step = resolve_arrivals(world, t)
        resolve_transfer_arrivals(world, t)

        shelter_urgency_estimates = {}
        for shelter in world.shelters.values():
            report_text = shelter.situation_report(rng)
            shelter_urgency_estimates[shelter.id] = field_report_agent.parse(report_text, t)

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

        shipped_this_step = {}
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
            desired_alloc = depot_agent.allocate(
                depot.id, depot.stock, transport_capacity, shelter_urgency_estimates, t,
                route_status=route_status,
                lookahead_context={
                    "timesteps_until_next_resupply": timesteps_until_next_resupply(world, depot),
                    "expected_resupply_fraction": depot.resupply_fraction,
                    "recent_avg_urgency_trend": list(urgency_history),
                },
                route_risk=route_risk)
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
                    # "shipped" here means "left the depot," matching this
                    # diagnostic's original utilization definition -- Phase R
                    # risk is resolved separately below and doesn't change
                    # what counts as shipped, only what actually arrives.
                    shipped_this_step[(depot.id, shelter_id, resource)] = qty
                    apply_shipment_with_risk(world, risk_rng, depot, shelter_id, resource, qty,
                                              vehicle_id, t, route_risk.get(shelter_id, 0.0))

        total_delivered = sum(shipped_this_step.values())
        urgency_history.append(round(average_urgency(shelter_urgency_estimates), 3))
        if len(urgency_history) > URGENCY_HISTORY_LENGTH:
            urgency_history.pop(0)
        trace.append({
            "config": config_name,
            "timestep": t,
            "total_available": round(total_available, 2),
            "total_delivered": round(total_delivered, 2),
            "delivered_pct_of_available": round(
                100.0 * total_delivered / total_available, 1) if total_available > 0 else 0.0,
            "pct_shelters_critical": round(critical_pct, 1),
            "unmet_need_this_step": round(unmet_urgent_need(world, arrived_this_step), 2),
        })
    return trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+",
                         default=["local_everywhere", "haiku_everywhere",
                                  "local_parsing_haiku_negotiation"],
                         choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--out", default="scarcity_diagnostics_trace.csv")
    args = parser.parse_args()

    cfg = dict(LA_WILDFIRES_JAN2025)
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    all_rows = []
    print("Running optimizer baseline trace...")
    all_rows += run_optimizer_with_trace(cfg)

    for config_name in args.configs:
        print(f"Running {config_name} trace...")
        all_rows += run_agent_config_with_trace(config_name, cfg, api_key)

    fieldnames = ["config", "timestep", "total_available", "total_delivered",
                  "delivered_pct_of_available", "pct_shelters_critical",
                  "unmet_need_this_step"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print()
    print("=" * 100)
    print("SUPPLY-CEILING DIAGNOSTIC SUMMARY")
    print("=" * 100)

    configs_present = ["optimizer_baseline"] + args.configs
    traces_by_config = {c: [r for r in all_rows if r["config"] == c] for c in configs_present}

    header = (f"{'config':<38} {'mean delivered/available %':>28} "
              f"{'mean %shelters critical':>26} {'t at first >=90% critical':>27}")
    print(header)
    print("-" * len(header))
    for c in configs_present:
        trace = traces_by_config[c]
        mean_delivered_pct = sum(r["delivered_pct_of_available"] for r in trace) / len(trace)
        mean_critical_pct = sum(r["pct_shelters_critical"] for r in trace) / len(trace)
        first_90 = next((r["timestep"] for r in trace if r["pct_shelters_critical"] >= 90.0), None)
        first_90_str = str(first_90) if first_90 is not None else "never"
        print(f"{c:<38} {mean_delivered_pct:>27.1f}% {mean_critical_pct:>25.1f}% "
              f"{first_90_str:>27}")

    print()
    print("How to read this:")
    print("  - If mean delivered/available % is close to 100% for EVERY config (including")
    print("    the optimizer), that confirms depots are maxed out almost every timestep")
    print("    regardless of which model is negotiating -- the supply-ceiling signature.")
    print("  - If mean %shelters critical is high (>=70-80%) and 'first >=90% critical'")
    print("    happens early (well before timestep 24) for every config, that confirms")
    print("    there's little room left for smart targeting to matter: almost everyone")
    print("    already qualifies as urgent, so redistributing the same limited pool")
    print("    across them barely changes the aggregate weighted shortfall.")
    print("  - If these numbers differ meaningfully BETWEEN configs, the supply-ceiling")
    print("    hypothesis is wrong (or incomplete) and the convergence needs a different")
    print("    explanation -- worth reporting honestly rather than forcing the narrative.")
    print(f"\nFull per-timestep trace written to {args.out}")


if __name__ == "__main__":
    main()