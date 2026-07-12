"""
Diagnoses *why* some multi-agent configs beat the "optimal" baseline on
cumulative unmet need, by logging per-timestep unmet need and total depot
stock remaining for both the optimizer and each multi-agent config, run
against identical seeded scenarios.

The hypothesis being tested: the optimizer is myopic (re-solves fresh each
timestep with no lookahead), so it may burn through depot stock aggressively
early and have nothing left for later, worse spikes -- while a multi-agent
config that happens to allocate more conservatively could end up "holding
reserves" that pay off later, purely by accident rather than by design.

If that's right, you should see:
  - The optimizer's depot stock hit near-zero earlier than the multi-agent
    configs that outperformed it.
  - Those outperforming configs' unmet-need trajectories look worse than the
    optimizer's in early timesteps, then better in later timesteps (i.e. a
    crossover), rather than being uniformly better throughout.

If leftover-stock patterns look similar across configs, this hypothesis is
wrong and something else is driving the result -- worth knowing before
writing either explanation into your report.

Usage:
    python run_diagnostics.py --timesteps 15 --seed 42 --configs local_everywhere haiku_everywhere
"""

from __future__ import annotations

import argparse
import copy
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


def total_depot_stock(world) -> float:
    return sum(sum(depot.stock.values()) for depot in world.depots.values())


def run_optimizer_with_trace(seed: int, timesteps: int) -> list[dict]:
    world = make_scenario(seed=seed)
    rng = random.Random(seed)
    trace = []
    for t in range(timesteps):
        advance_disaster(world, rng)
        alloc = solve_optimal_allocation(world)
        for (depot_id, shelter_id, resource), qty in alloc.items():
            world.depots[depot_id].stock[resource] = max(
                0.0, world.depots[depot_id].stock.get(resource, 0.0) - qty)
        step_unmet = unmet_urgent_need(world, alloc)
        trace.append({
            "config": "optimizer_baseline",
            "timestep": t,
            "unmet_need_this_step": round(step_unmet, 2),
            "total_depot_stock_remaining": round(total_depot_stock(world), 2),
            "max_shelter_urgency": round(
                max(s.urgency(r) for s in world.shelters.values() for r in RESOURCE_TYPES), 2),
        })
    return trace


def run_agents_with_trace(seed: int, timesteps: int, config_name: str,
                            api_key: str) -> list[dict]:
    fr_model, depot_model, dispatch_model = MODEL_CONFIGS[config_name]
    tracker = CostTracker()
    world = make_scenario(seed=seed)
    rng = random.Random(seed)
    # RNG STREAM SEPARATION -- see FINDINGS.md "RNG Stream Contamination"
    # and run_simulation.py's run_multi_agent_system for the full
    # explanation. Derived immediately, before any config-dependent draws.
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
            lookahead_context = {
                "timesteps_until_next_resupply": timesteps_until_next_resupply(world, depot),
                "expected_resupply_fraction": depot.resupply_fraction,
                "recent_avg_urgency_trend": list(urgency_history),
            }
            desired_alloc = depot_agent.allocate(
                depot.id, depot.stock, transport_capacity,
                shelter_urgency_estimates, t, route_status=route_status,
                lookahead_context=lookahead_context, route_risk=route_risk,
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
                    apply_shipment_with_risk(world, risk_rng, depot, shelter_id, resource, qty,
                                              vehicle_id, t, route_risk.get(shelter_id, 0.0))

        step_unmet = unmet_urgent_need(world, arrived_this_step)
        urgency_history.append(round(average_urgency(shelter_urgency_estimates), 3))
        if len(urgency_history) > URGENCY_HISTORY_LENGTH:
            urgency_history.pop(0)
        trace.append({
            "config": config_name,
            "timestep": t,
            "unmet_need_this_step": round(step_unmet, 2),
            "total_depot_stock_remaining": round(total_depot_stock(world), 2),
            "max_shelter_urgency": round(
                max(s.urgency(r) for s in world.shelters.values() for r in RESOURCE_TYPES), 2),
        })
    return trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--configs", nargs="+", default=list(MODEL_CONFIGS.keys()),
                         choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--out", default="diagnostics_trace.csv")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")

    all_rows = []
    print("Running optimizer baseline trace...")
    all_rows += run_optimizer_with_trace(args.seed, args.timesteps)

    for config_name in args.configs:
        print(f"Running {config_name} trace...")
        all_rows += run_agents_with_trace(args.seed, args.timesteps, config_name, api_key)

    fieldnames = ["config", "timestep", "unmet_need_this_step",
                  "total_depot_stock_remaining", "max_shelter_urgency"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # Print the specific diagnostic: final depot stock leftover, and whether
    # unmet-need trajectories cross over (worse early, better late) for each
    # config relative to the optimizer -- the signature of the hoarding
    # hypothesis, as opposed to uniformly-better-or-worse trajectories.
    print()
    print("=" * 90)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 90)

    configs_present = ["optimizer_baseline"] + args.configs
    traces_by_config = {c: [r for r in all_rows if r["config"] == c] for c in configs_present}

    print(f"{'config':<38} {'final_stock':>12} {'early_unmet(0-4)':>17} {'late_unmet(11-14)':>18}")
    print("-" * 90)
    for c in configs_present:
        trace = traces_by_config[c]
        final_stock = trace[-1]["total_depot_stock_remaining"]
        early = sum(r["unmet_need_this_step"] for r in trace if r["timestep"] < 5)
        late = sum(r["unmet_need_this_step"] for r in trace if r["timestep"] >= args.timesteps - 4)
        print(f"{c:<38} {final_stock:>12.1f} {early:>17.1f} {late:>18.1f}")

    print()
    print("How to read this:")
    print("  - If a config has LOWER final_stock than the optimizer, it spent more")
    print("    aggressively (opposite of the hoarding hypothesis for that config).")
    print("  - If a config has HIGHER final_stock AND lower late_unmet than the")
    print("    optimizer, that's the hoarding-pays-off signature: it held reserves")
    print("    and cashed them in when things got worse later.")
    print("  - If a config has similar final_stock to the optimizer but still beat")
    print("    it on cumulative unmet need, the hoarding hypothesis is wrong --")
    print("    something else (e.g. how urgency estimates drove targeting) is the")
    print("    real explanation, and is worth investigating next.")
    print()
    print(f"Full per-timestep trace written to {args.out}")


if __name__ == "__main__":
    main()