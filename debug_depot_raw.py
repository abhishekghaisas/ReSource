"""
Confirms (or refutes) the hypothesis that Sonnet's Depot Agent negotiation
is failing because its raw response doesn't survive the JSON-extraction
regex in agents/depot_agent.py -- rather than Sonnet genuinely "choosing"
to allocate nothing.

Prints the RAW, unparsed model output for every depot negotiation call in
the first several timesteps, alongside what our parser extracted from it.
If raw output contains real allocation numbers but "parsed" comes back
empty, that confirms a parsing bug, not a negotiation strategy.

Usage:
    python debug_depot_raw.py --config sonnet_everywhere --timesteps 6
"""

from __future__ import annotations

import argparse
import os
import random

from agents.base_agent import CostTracker, LLMBackend
from agents.depot_agent import DepotAgent
from agents.field_report_agent import FieldReportAgent
from run_simulation import MODEL_CONFIGS
from simulation.world import advance_disaster, make_scenario


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="sonnet_everywhere", choices=list(MODEL_CONFIGS))
    parser.add_argument("--timesteps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    fr_model, depot_model, _ = MODEL_CONFIGS[args.config]

    world = make_scenario(seed=args.seed)
    rng = random.Random(args.seed)
    tracker = CostTracker()

    field_report_agent = FieldReportAgent(LLMBackend(fr_model, tracker, "field_report", api_key))
    depot_backend = LLMBackend(depot_model, tracker, "depot_negotiation", api_key)

    for t in range(args.timesteps):
        advance_disaster(world, rng)

        shelter_urgency_estimates = {}
        for shelter in world.shelters.values():
            report_text = shelter.situation_report(rng)
            shelter_urgency_estimates[shelter.id] = field_report_agent.parse(report_text, t)

        depot = list(world.depots.values())[0]
        transport_capacity = sum(tr.capacity for tr in world.transports.values())

        # Call the raw backend directly (bypassing DepotAgent's parsing) so we
        # can see exactly what the model said before any extraction happens.
        import json
        prompt = json.dumps({
            "depot_id": depot.id,
            "stock": depot.stock,
            "transport_capacity": transport_capacity,
            "shelter_urgency_reports": shelter_urgency_estimates,
        }, indent=2)

        from agents.depot_agent import SYSTEM_PROMPT, DepotAgent as DA
        # Match the real max_tokens used in depot_agent.py's allocate() --
        # this was hardcoded lower here before, which meant this debug
        # script was silently re-testing the ORIGINAL bug instead of the fix.
        raw = depot_backend.call(SYSTEM_PROMPT, prompt, timestep=t, max_tokens=1500)
        parsed = DA._extract_and_validate(raw, depot.stock, transport_capacity, shelter_urgency_estimates)

        print("=" * 90)
        print(f"TIMESTEP {t}")
        print("-" * 90)
        print("RAW MODEL OUTPUT:")
        print(raw)
        print("-" * 90)
        print("PARSED/CLAMPED ALLOCATION:", parsed)
        print()

        # Apply the (possibly empty) allocation so subsequent timesteps see
        # realistic depleting stock, matching the real run's behavior.
        for shelter_id, resources in parsed.items():
            for resource, qty in resources.items():
                depot.stock[resource] = max(0.0, depot.stock.get(resource, 0.0) - qty)


if __name__ == "__main__":
    main()