"""
Directly verifies Phase A (real transit delay) against REAL model output,
not a stub -- confirms shipments actually take >0 timesteps to arrive and
vehicles actually go busy, without the severe-scarcity confound that made
the full real-disaster scenario's aggregate unmet-need trace hard to read
for this specific question.

Uses a small, cheap toy-scale scenario (not the LA-wildfire calibration) so
this is fast and, with --config local_everywhere, free.

Usage:
    python run_shipment_trace.py
    python run_shipment_trace.py --config local_parsing_haiku_negotiation
"""

from __future__ import annotations

import argparse
import os
import random

from agents.base_agent import CostTracker, LLMBackend
from agents.depot_agent import DepotAgent
from agents.dispatcher_agent import DispatcherAgent
from agents.field_report_agent import FieldReportAgent
from run_simulation import MODEL_CONFIGS
from simulation.world import advance_disaster, dispatch_shipment, make_scenario, resolve_arrivals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="local_everywhere", choices=list(MODEL_CONFIGS))
    parser.add_argument("--timesteps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--debug", action="store_true",
                         help="Print raw model output whenever DepotAgent's allocation or "
                              "DispatcherAgent's vehicle assignment comes back empty, to "
                              "distinguish 'nothing needed shipping' from 'the model's output "
                              "failed to parse.'")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    fr_model, depot_model, dispatch_model = MODEL_CONFIGS[args.config]
    tracker = CostTracker()

    if args.debug:
        # Wrap LLMBackend.call to print the RAW response before any parsing
        # happens -- the cleanest way to see whether a model's output failed
        # to parse (garbled/non-JSON text) versus legitimately decided to
        # allocate/dispatch nothing this round.
        original_call = LLMBackend.call

        def debug_call(self, system_prompt, user_prompt, timestep, max_tokens=300):
            text = original_call(self, system_prompt, user_prompt, timestep, max_tokens)
            preview = text.strip().replace("\n", " ")[:300]
            print(f"    [DEBUG t={timestep} role={self.agent_role} model={self.model}] "
                  f"raw_response={preview!r}")
            return text

        LLMBackend.call = debug_call

    world = make_scenario(seed=args.seed, disaster_type="flood", n_shelters=3,
                           n_depots=1, n_transports=3, grid_size=(6, 6))
    rng = random.Random(args.seed)

    field_report_agent = FieldReportAgent(LLMBackend(fr_model, tracker, "field_report", api_key))
    depot_agent = DepotAgent(LLMBackend(depot_model, tracker, "depot_negotiation", api_key))
    dispatcher_agent = DispatcherAgent(LLMBackend(dispatch_model, tracker, "dispatcher", api_key))

    print(f"config={args.config}  seed={args.seed}  timesteps={args.timesteps}")
    print(f"{'t':>2}  {'dispatched_this_step':>20}  {'in_transit_after':>17}  "
          f"{'arrived_this_step':>18}  {'vehicles_busy':>13}")
    print("-" * 80)

    for t in range(args.timesteps):
        advance_disaster(world, rng)
        arrived = resolve_arrivals(world, t)
        arrived_total = sum(arrived.values())

        shelter_urgency_estimates = {}
        for shelter in world.shelters.values():
            report_text = shelter.situation_report(rng)
            shelter_urgency_estimates[shelter.id] = field_report_agent.parse(report_text, t)

        dispatched_this_step = 0.0
        for depot in world.depots.values():
            transport_capacity = sum(
                tr.capacity for tr in world.transports.values() if tr.busy_until <= t
            )
            route_status = {
                shelter.id: not world.route_damaged(depot.position, shelter.position)
                for shelter in world.shelters.values()
            }
            desired_alloc = depot_agent.allocate(
                depot.id, depot.stock, transport_capacity, shelter_urgency_estimates, t,
                route_status=route_status)
            vehicle_assignments = dispatcher_agent.sequence(
                world, depot.id, desired_alloc, shelter_urgency_estimates, t)
            for vehicle_id, deliveries in vehicle_assignments.items():
                for delivery in deliveries:
                    qty = delivery["quantity"]
                    if qty <= 0:
                        continue
                    resource = delivery["resource"]
                    shelter_id = delivery["shelter_id"]
                    depot.stock[resource] = max(0.0, depot.stock.get(resource, 0.0) - qty)
                    dispatch_shipment(world, depot.id, shelter_id, resource, qty, vehicle_id, t)
                    dispatched_this_step += qty

        vehicles_busy = sum(1 for tr in world.transports.values() if tr.busy_until > t)
        print(f"{t:>2}  {dispatched_this_step:>20.1f}  {len(world.in_transit):>17}  "
              f"{arrived_total:>18.1f}  {vehicles_busy:>13}")

    print()
    print("How to read this:")
    print("  - 'dispatched_this_step' > 0 with 'arrived_this_step' = 0 in the SAME row is the")
    print("    core thing we're checking: a delivery just sent out has NOT arrived yet.")
    print("  - A later row's 'arrived_this_step' > 0 should correspond to an earlier row's")
    print("    dispatch, offset by that shipment's travel time -- not the same timestep.")
    print("  - 'vehicles_busy' > 0 right after a dispatch, dropping back down once the round")
    print("    trip completes, confirms Transport.busy_until is actually being enforced.")
    print()
    print(tracker.summary())


if __name__ == "__main__":
    main()