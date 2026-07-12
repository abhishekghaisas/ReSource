"""
Originally diagnosed WHY wildfire was the one scenario where every config
(local, Haiku, mixed) lost to the optimizer baseline, when the same configs
beat it comfortably on flood. That was traced to DepotAgent having no
route-awareness, which has since been fixed (Phase B: DepotAgent now
receives route_status per shelter and masks blocked ones internally) and
further hardened (Phase A: the Dispatcher independently re-checks
route_damaged when actually assigning a vehicle, and stock is only deducted
at that real ship time, not at DepotAgent's decision time).

This script now serves as an ongoing regression check for both defenses:
it measures "wasted stock" -- shipped quantity that ends up going to a
route that's blocked at actual ship time -- which should read ~0 for every
config, matching the optimizer (which cannot waste stock this way by
construction). A nonzero reading here means one of the two route-awareness
defenses has regressed, not that the original hypothesis was right after
all (see FINDINGS.md Issue #4 for that history).

Usage:
    python run_wildfire_diagnostics.py --timesteps 15 --seed 7
    python run_wildfire_diagnostics.py --timesteps 15 --seed 7 \
        --disaster-type flood        # run as a control comparison
    python run_wildfire_diagnostics.py --timesteps 15 --seed 7 \
        --configs local_everywhere haiku_everywhere
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
from run_simulation import (MODEL_CONFIGS, URGENCY_HISTORY_LENGTH, average_urgency,
                             build_depot_summaries, compute_route_risk_for_depot,
                             apply_shipment_with_risk, apply_transfer_with_risk,
                             rotate_depots_for_fairness)
from simulation.world import (RESOURCE_TYPES, advance_disaster, dispatch_shipment,
                               dispatch_transfer, make_scenario, resolve_arrivals,
                               resolve_transfer_arrivals, route_risk_score,
                               timesteps_until_next_resupply)

WILDFIRE_SPREAD_RATE_DEFAULT = 0.6  # must match simulation.world.advance_disaster's default


def total_depot_stock(world) -> float:
    return sum(sum(depot.stock.values()) for depot in world.depots.values())


def fire_radius_at(world, wildfire_spread_rate: float = WILDFIRE_SPREAD_RATE_DEFAULT) -> float:
    if world.disaster_type != "wildfire":
        return float("nan")
    return wildfire_spread_rate * world.timestep


def run_optimizer_with_trace(seed: int, timesteps: int, disaster_type: str) -> list[dict]:
    world = make_scenario(seed=seed, disaster_type=disaster_type)
    rng = random.Random(seed)
    trace = []
    for t in range(timesteps):
        advance_disaster(world, rng)
        alloc = solve_optimal_allocation(world)

        # Sanity check: the optimizer should NEVER allocate anything to a
        # blocked route, since it's zeroed out in its own bounds. If this is
        # ever nonzero, that's a bug in the optimizer, not a finding.
        wasted = 0.0
        for (depot_id, shelter_id, resource), qty in alloc.items():
            depot = world.depots[depot_id]
            shelter = world.shelters[shelter_id]
            if qty > 0 and world.route_damaged(depot.position, shelter.position):
                wasted += qty

        for (depot_id, shelter_id, resource), qty in alloc.items():
            world.depots[depot_id].stock[resource] = max(
                0.0, world.depots[depot_id].stock.get(resource, 0.0) - qty)
        step_unmet = unmet_urgent_need(world, alloc)
        resupply_status = ",".join(f"{did}:{status}" for did, status in world.last_resupply.items())
        trace.append({
            "config": "optimizer_baseline",
            "disaster_type": disaster_type,
            "timestep": t,
            "fire_radius": round(fire_radius_at(world), 2),
            "blocked_routes_count": len(world.blocked_routes),
            "resupply_status": resupply_status,
            "unmet_need_this_step": round(step_unmet, 2),
            "wasted_stock_this_step": round(wasted, 2),
            "total_depot_stock_remaining": round(total_depot_stock(world), 2),
            "max_shelter_urgency": round(
                max(s.urgency(r) for s in world.shelters.values() for r in RESOURCE_TYPES), 2),
        })
    return trace


def run_agents_with_trace(seed: int, timesteps: int, config_name: str, disaster_type: str,
                           api_key: str) -> list[dict]:
    fr_model, depot_model, dispatch_model = MODEL_CONFIGS[config_name]
    tracker = CostTracker()
    world = make_scenario(seed=seed, disaster_type=disaster_type)
    rng = random.Random(seed)
    # RNG STREAM SEPARATION -- see FINDINGS.md "RNG Stream Contamination".
    risk_rng = random.Random(rng.random())

    field_report_agent = FieldReportAgent(
        LLMBackend(fr_model, tracker, "field_report", api_key))
    depot_agent = DepotAgent(
        LLMBackend(depot_model, tracker, "depot_negotiation", api_key))
    dispatcher_agent = DispatcherAgent(
        LLMBackend(dispatch_model, tracker, "dispatcher", api_key))
    inter_depot_agent = InterDepotCoordinatorAgent(
        LLMBackend(depot_model, tracker, "inter_depot_coordination", api_key))

    trace = []
    urgency_history = []
    for t in range(timesteps):
        advance_disaster(world, rng)

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

        wasted = 0.0
        for depot in rotate_depots_for_fairness(world, t):
            transport_capacity = sum(
                tr.capacity for tr in world.transports.values()
                if tr.busy_until <= t and not tr.destroyed
            )
            # UPDATED (Phase B): DepotAgent now receives route_status per
            # shelter and masks out blocked ones internally before a model's
            # allocation can even be considered (see agents/depot_agent.py).
            # UPDATED (Phase A): stock is no longer deducted at negotiation
            # time -- only at actual ship time, once the Dispatcher assigns
            # a real vehicle (which ALSO re-checks route_damaged as a second,
            # independent defense -- see agents/dispatcher_agent.py). So
            # `wasted` below is now checked against SHIPPED quantities, not
            # DepotAgent's raw allocation -- with two independent route
            # checks in the pipeline, this should read ~0 for every agent
            # config, matching the optimizer. If it doesn't, that's a
            # regression in one of those two defenses, not evidence the
            # original route-blindness hypothesis was right after all (that
            # hypothesis was tested and falsified before either feature
            # existed; see FINDINGS.md Issue #4). Phase R note: this
            # "wasted" tracker is about HARD-blocked routes only -- a
            # partial/total loss on a RISKY-but-not-blocked route is a
            # different, intentional mechanic (see risk_outcomes below),
            # not evidence of the same route-blindness bug.
            route_status = {
                shelter.id: not world.route_damaged(depot.position, shelter.position)
                for shelter in world.shelters.values()
            }
            route_risk = compute_route_risk_for_depot(world, depot)
            desired_alloc = depot_agent.allocate(
                depot.id, depot.stock, transport_capacity,
                shelter_urgency_estimates, t, route_status=route_status,
                lookahead_context={
                    "timesteps_until_next_resupply": timesteps_until_next_resupply(world, depot),
                    "expected_resupply_fraction": depot.resupply_fraction,
                    "recent_avg_urgency_trend": list(urgency_history),
                },
                route_risk=route_risk,
            )
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
                    shelter = world.shelters[shelter_id]
                    route_blocked = world.route_damaged(depot.position, shelter.position)
                    apply_shipment_with_risk(world, risk_rng, depot, shelter_id, resource, qty,
                                              vehicle_id, t, route_risk.get(shelter_id, 0.0))
                    if route_blocked:
                        wasted += qty

        step_unmet = unmet_urgent_need(world, arrived_this_step)
        urgency_history.append(round(average_urgency(shelter_urgency_estimates), 3))
        if len(urgency_history) > URGENCY_HISTORY_LENGTH:
            urgency_history.pop(0)
        resupply_status = ",".join(f"{did}:{status}" for did, status in world.last_resupply.items())
        trace.append({
            "config": config_name,
            "disaster_type": disaster_type,
            "timestep": t,
            "fire_radius": round(fire_radius_at(world), 2),
            "blocked_routes_count": len(world.blocked_routes),
            "resupply_status": resupply_status,
            "unmet_need_this_step": round(step_unmet, 2),
            "wasted_stock_this_step": round(wasted, 2),
            "total_depot_stock_remaining": round(total_depot_stock(world), 2),
            "max_shelter_urgency": round(
                max(s.urgency(r) for s in world.shelters.values() for r in RESOURCE_TYPES), 2),
        })
    return trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--disaster-type", default="wildfire",
                         choices=["flood", "wildfire", "earthquake"],
                         help="Defaults to wildfire since that's the scenario "
                              "under investigation. Pass --disaster-type flood "
                              "to run the same diagnostic as a control.")
    parser.add_argument("--configs", nargs="+", default=list(MODEL_CONFIGS.keys()),
                         choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--out", default="wildfire_diagnostics_trace.csv")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")

    all_rows = []
    print(f"Running optimizer baseline trace ({args.disaster_type}, seed={args.seed})...")
    all_rows += run_optimizer_with_trace(args.seed, args.timesteps, args.disaster_type)

    for config_name in args.configs:
        print(f"Running {config_name} trace...")
        try:
            all_rows += run_agents_with_trace(args.seed, args.timesteps, config_name,
                                               args.disaster_type, api_key)
        except Exception as e:
            print(f"  FAILED: {e}")

    fieldnames = ["config", "disaster_type", "timestep", "fire_radius",
                  "blocked_routes_count", "resupply_status", "unmet_need_this_step",
                  "wasted_stock_this_step", "total_depot_stock_remaining",
                  "max_shelter_urgency"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print()
    print("=" * 100)
    print(f"DIAGNOSTIC SUMMARY  (disaster_type={args.disaster_type}, seed={args.seed})")
    print("=" * 100)

    configs_present = ["optimizer_baseline"] + [c for c in args.configs
                                                  if any(r["config"] == c for r in all_rows)]
    traces_by_config = {c: [r for r in all_rows if r["config"] == c] for c in configs_present}

    header = f"{'config':<38} {'total_wasted_stock':>19} {'cumulative_unmet':>17} {'final_blocked_routes':>21}"
    print(header)
    print("-" * len(header))
    for c in configs_present:
        trace = traces_by_config[c]
        total_wasted = sum(r["wasted_stock_this_step"] for r in trace)
        cumulative_unmet = sum(r["unmet_need_this_step"] for r in trace)
        final_blocked = trace[-1]["blocked_routes_count"]
        print(f"{c:<38} {total_wasted:>19.1f} {cumulative_unmet:>17.1f} {final_blocked:>21}")

    print()
    print("How to read this:")
    print("  - optimizer_baseline's total_wasted_stock should be ~0 (it zeroes out")
    print("    blocked routes before optimizing -- if nonzero, that's a bug worth")
    print("    reporting separately, not part of this hypothesis).")
    print("  - If an agent config's total_wasted_stock is LARGE relative to the")
    print("    optimizer's, that's direct evidence DepotAgent's lack of route-")
    print("    awareness is burning stock on shelters it can no longer reach --")
    print("    the leading hypothesis for why wildfire uniquely hurts every config.")
    print("  - Compare this same run with --disaster-type flood as a control: if")
    print("    wasted_stock is much smaller there, that confirms it's wildfire's")
    print("    concentrated, escalating route loss (not agent competence generally)")
    print("    that's driving the effect.")
    print()
    print(f"Full per-timestep trace written to {args.out}")


if __name__ == "__main__":
    main()