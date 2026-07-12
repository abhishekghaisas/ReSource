"""
Tests the "dispatch is the new bottleneck" hypothesis: local_everywhere and
local_parsing_haiku_negotiation converged to near-identical outcomes
(+7.7% vs +7.7%, nearly identical cumulative_unmet at every timestep) on
the real-disaster scenario under Phase A, despite one using Haiku for
negotiation and the other using the local model for everything. Both
configs use the LOCAL model for DISPATCH, though -- MODEL_CONFIGS[
"local_parsing_haiku_negotiation"] = ("local", "claude-haiku...", "local").
Before Phase A, dispatch decisions were discarded entirely, so this
didn't matter. Now that dispatch actually controls what ships, if the
local dispatcher is the real bottleneck, better negotiation upstream
could be getting capped by weaker dispatch downstream regardless.

WHAT THIS MEASURES, PER TIMESTEP, PER CONFIG
-------------------------------------------------------------------------
  1. total_desired: sum of DepotAgent's allocation decision (before
     dispatch) -- this is what DIFFERS between local and Haiku negotiation
     if the negotiators are actually deciding differently.
  2. total_shipped: sum of what DispatcherAgent actually assigns to a
     vehicle -- this is capped by dispatch quality/vehicle availability,
     which is LOCAL in both configs being compared here.
  3. shipped_pct_of_desired: how much of what was decided actually made it
     onto a vehicle. If this is similarly low for both configs even though
     total_desired differs a lot, that's the smoking gun -- the dispatcher
     is throwing away the benefit of better negotiation.
  4. vehicles_used / vehicles_available: are available vehicles actually
     being utilized, or sitting idle because the dispatcher failed to
     assign them (parsing failure, hallucinated unavailable vehicle, etc.)?

Usage:
    python run_dispatch_bottleneck_diagnostics.py
    python run_dispatch_bottleneck_diagnostics.py --timesteps 12 --seed 2025
"""

from __future__ import annotations

import argparse
import os
import random

from agents.base_agent import CostTracker, LLMBackend
from agents.depot_agent import DepotAgent
from agents.dispatcher_agent import DispatcherAgent
from agents.field_report_agent import FieldReportAgent
from run_real_disaster import (
    LA_WILDFIRES_JAN2025,
    apply_disease_outbreak,
    apply_population_surge,
)
from run_simulation import MODEL_CONFIGS
from simulation.world import advance_disaster, dispatch_shipment, make_scenario, resolve_arrivals

CONFIGS_TO_COMPARE = ["local_everywhere", "local_parsing_haiku_negotiation"]


def build_world(cfg: dict):
    world = make_scenario(
        seed=cfg["seed"], disaster_type="wildfire",
        n_shelters=cfg["n_shelters"], n_depots=cfg["n_depots"],
        n_transports=cfg["n_transports"], grid_size=cfg["grid_size"],
    )
    apply_population_surge(world, cfg["population_surge_factor"])
    return world


def run_config_trace(config_name: str, cfg: dict, api_key: str) -> list[dict]:
    fr_model, depot_model, dispatch_model = MODEL_CONFIGS[config_name]
    tracker = CostTracker()
    world = build_world(cfg)
    rng = random.Random(cfg["seed"])

    field_report_agent = FieldReportAgent(LLMBackend(fr_model, tracker, "field_report", api_key))
    depot_agent = DepotAgent(LLMBackend(depot_model, tracker, "depot_negotiation", api_key))
    dispatcher_agent = DispatcherAgent(LLMBackend(dispatch_model, tracker, "dispatcher", api_key))

    trace = []
    outbreak_applied = False
    for t in range(cfg["timesteps"]):
        if not outbreak_applied and t >= cfg["disease_outbreak_timestep"]:
            apply_disease_outbreak(world, cfg["disease_outbreak_medical_multiplier"])
            outbreak_applied = True
        advance_disaster(world, rng, wildfire_spread_rate=cfg["wildfire_spread_rate"])
        resolve_arrivals(world, t)  # not used for this diagnostic's metrics, just kept consistent

        shelter_urgency_estimates = {}
        for shelter in world.shelters.values():
            report_text = shelter.situation_report(rng)
            shelter_urgency_estimates[shelter.id] = field_report_agent.parse(report_text, t)

        total_desired = 0.0
        total_shipped = 0.0
        vehicles_available_total = 0
        vehicles_used_total = 0

        for depot in world.depots.values():
            transport_capacity = sum(
                tr.capacity for tr in world.transports.values() if tr.busy_until <= t
            )
            vehicles_available_total += sum(
                1 for tr in world.transports.values() if tr.busy_until <= t
            )
            route_status = {
                shelter.id: not world.route_damaged(depot.position, shelter.position)
                for shelter in world.shelters.values()
            }
            desired_alloc = depot_agent.allocate(
                depot.id, depot.stock, transport_capacity, shelter_urgency_estimates, t,
                route_status=route_status)
            total_desired += sum(sum(r.values()) for r in desired_alloc.values())

            vehicle_assignments = dispatcher_agent.sequence(
                world, depot.id, desired_alloc, shelter_urgency_estimates, t)
            vehicles_used_total += len(vehicle_assignments)
            for vehicle_id, deliveries in vehicle_assignments.items():
                for delivery in deliveries:
                    qty = delivery["quantity"]
                    if qty <= 0:
                        continue
                    resource = delivery["resource"]
                    shelter_id = delivery["shelter_id"]
                    depot.stock[resource] = max(0.0, depot.stock.get(resource, 0.0) - qty)
                    dispatch_shipment(world, depot.id, shelter_id, resource, qty, vehicle_id, t)
                    total_shipped += qty

        shipped_pct = (100.0 * total_shipped / total_desired) if total_desired > 0 else 0.0
        vehicle_util_pct = (100.0 * vehicles_used_total / vehicles_available_total
                             if vehicles_available_total > 0 else 0.0)
        trace.append({
            "config": config_name,
            "timestep": t,
            "total_desired": round(total_desired, 1),
            "total_shipped": round(total_shipped, 1),
            "shipped_pct_of_desired": round(shipped_pct, 1),
            "vehicles_available": vehicles_available_total,
            "vehicles_used": vehicles_used_total,
            "vehicle_util_pct": round(vehicle_util_pct, 1),
        })
    return trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    cfg = dict(LA_WILDFIRES_JAN2025)
    cfg["timesteps"] = args.timesteps
    cfg["seed"] = args.seed
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    all_traces = {}
    for config_name in CONFIGS_TO_COMPARE:
        print(f"Running {config_name}...")
        all_traces[config_name] = run_config_trace(config_name, cfg, api_key)

    print()
    print("=" * 100)
    print("DISPATCH-BOTTLENECK DIAGNOSTIC SUMMARY")
    print("=" * 100)
    header = (f"{'config':<32} {'mean_total_desired':>19} {'mean_total_shipped':>19} "
              f"{'mean_shipped_pct':>17} {'mean_vehicle_util_pct':>22}")
    print(header)
    print("-" * len(header))
    for config_name, trace in all_traces.items():
        mean_desired = sum(r["total_desired"] for r in trace) / len(trace)
        mean_shipped = sum(r["total_shipped"] for r in trace) / len(trace)
        mean_shipped_pct = sum(r["shipped_pct_of_desired"] for r in trace) / len(trace)
        mean_util = sum(r["vehicle_util_pct"] for r in trace) / len(trace)
        print(f"{config_name:<32} {mean_desired:>19.1f} {mean_shipped:>19.1f} "
              f"{mean_shipped_pct:>16.1f}% {mean_util:>21.1f}%")

    print()
    print("How to read this:")
    print("  - If 'mean_total_desired' differs meaningfully between configs (Haiku vs local")
    print("    negotiating differently) but 'mean_total_shipped' is similar anyway, that's the")
    print("    smoking gun: the (local, shared) dispatcher is capping the outcome regardless")
    print("    of how good the upstream negotiation was.")
    print("  - Low 'mean_vehicle_util_pct' for both configs (available vehicles sitting idle)")
    print("    points at dispatch parsing/decision failures, not a lack of things to ship.")
    print("  - If both desired AND shipped are similar across configs, the negotiators")
    print("    themselves may be converging (less likely given Haiku vs local, but worth")
    print("    checking rather than assuming).")


if __name__ == "__main__":
    main()