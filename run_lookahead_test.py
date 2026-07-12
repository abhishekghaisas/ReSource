"""
Directly tests whether DepotAgent's lookahead_context (Phase D) actually
changes real-model behavior, rather than inferring it indirectly from a
full scenario run where scarcity/route effects would confound the signal
(the same mistake made testing Phase A's transit delay the first time --
see project history).

Holds stock, shelter requests, and route_status IDENTICAL across two
cases, varying ONLY lookahead_context, and compares total allocated
quantity:

  Case A ("spend freely"): resupply arrives next round (timesteps_until_
  next_resupply=0), urgency trend flat/stable. A sensible negotiator
  should have less reason to hold back -- more is coming imminently.

  Case B ("hold back"): resupply is far away (10 rounds), urgency trend
  sharply rising. A sensible negotiator has more reason to reserve some
  stock for a worse round ahead, rather than spend it all now.

This does NOT test "which total is more correct" (both could be
reasonable) -- it tests whether the two cases produce DIFFERENT behavior
in the expected direction, which is the actual claim Phase D makes: that
lookahead context is being used at all, not ignored.

Usage:
    python run_lookahead_test.py --config local_parsing_haiku_negotiation_dispatch --repeats 3
    python run_lookahead_test.py --config haiku_everywhere --repeats 3
"""

from __future__ import annotations

import argparse
import os

from agents.base_agent import CostTracker, LLMBackend
from agents.depot_agent import DepotAgent
from run_simulation import MODEL_CONFIGS

STOCK = {"food": 20.0, "water": 20.0, "medical": 20.0}
TRANSPORT_CAPACITY = 100.0  # generous -- not meant to be the binding constraint here
SHELTER_REQUESTS = {
    "S1": {"food": 0.6, "water": 0.6, "medical": 0.6, "population_estimate": 100},
    "S2": {"food": 0.6, "water": 0.6, "medical": 0.6, "population_estimate": 100},
    "S3": {"food": 0.6, "water": 0.6, "medical": 0.6, "population_estimate": 100},
}
ROUTE_STATUS = {sid: True for sid in SHELTER_REQUESTS}

CASE_A_SPEND_FREELY = {
    "timesteps_until_next_resupply": 0,
    "expected_resupply_fraction": 0.5,
    "recent_avg_urgency_trend": [0.5, 0.5, 0.5],
}
CASE_B_HOLD_BACK = {
    "timesteps_until_next_resupply": 10,
    "expected_resupply_fraction": 0.5,
    "recent_avg_urgency_trend": [0.3, 0.5, 0.8],
}


def total_allocated(alloc: dict) -> float:
    return sum(qty for resources in alloc.values() for qty in resources.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+",
                         default=["local_everywhere", "haiku_everywhere",
                                  "local_parsing_haiku_negotiation_dispatch"],
                         choices=list(MODEL_CONFIGS))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")

    # Note which configs share a negotiation model up front -- e.g.
    # haiku_everywhere and local_parsing_haiku_negotiation_dispatch both
    # select Haiku for depot_model, so for THIS test (which only ever calls
    # DepotAgent directly) they're expected to behave identically up to
    # sampling noise, not as an independent confirmation.
    model_to_configs: dict = {}
    for c in args.configs:
        depot_model = MODEL_CONFIGS[c][1]
        model_to_configs.setdefault(depot_model, []).append(c)
    for model, configs in model_to_configs.items():
        if len(configs) > 1:
            print(f"NOTE: {', '.join(configs)} all use '{model}' for negotiation -- "
                  f"expect near-identical results between them here, not independent evidence.")
    print()

    all_summaries = {}

    for config_name in args.configs:
        depot_model = MODEL_CONFIGS[config_name][1]
        print(f"=== {config_name}  (negotiation model: {depot_model}) ===")

        results_a, results_b = [], []
        for rep in range(args.repeats):
            tracker = CostTracker()
            depot_agent = DepotAgent(LLMBackend(depot_model, tracker, "depot_negotiation", api_key))

            alloc_a = depot_agent.allocate(
                "D1", dict(STOCK), TRANSPORT_CAPACITY, SHELTER_REQUESTS, timestep=0,
                route_status=ROUTE_STATUS, lookahead_context=CASE_A_SPEND_FREELY)
            total_a = total_allocated(alloc_a)
            results_a.append(total_a)

            alloc_b = depot_agent.allocate(
                "D1", dict(STOCK), TRANSPORT_CAPACITY, SHELTER_REQUESTS, timestep=0,
                route_status=ROUTE_STATUS, lookahead_context=CASE_B_HOLD_BACK)
            total_b = total_allocated(alloc_b)
            results_b.append(total_b)

            print(f"  repeat {rep+1}/{args.repeats}: "
                  f"Case A (spend freely)={total_a:.1f}   Case B (hold back)={total_b:.1f}")

        mean_a = sum(results_a) / len(results_a)
        mean_b = sum(results_b) / len(results_b)
        all_summaries[config_name] = (mean_a, mean_b)
        print(f"  SUMMARY: Case A mean={mean_a:.2f}   Case B mean={mean_b:.2f}   "
              f"delta={mean_a - mean_b:+.2f}\n")

    print("=" * 78)
    print("CROSS-CONFIG SUMMARY")
    print("=" * 78)
    print(f"{'config':<42} {'Case A mean':>12} {'Case B mean':>12} {'delta':>10}")
    print("-" * 78)
    for config_name, (mean_a, mean_b) in all_summaries.items():
        print(f"{config_name:<42} {mean_a:>12.2f} {mean_b:>12.2f} {mean_a - mean_b:>+10.2f}")
    print()
    print("How to read this:")
    print("  - Positive delta = allocates MORE when resupply is imminent/urgency stable")
    print("    (Case A) than when resupply is far/urgency rising (Case B) -- matches the")
    print("    intended reasoning behind lookahead_context.")
    print("  - Compare local_everywhere's delta against the Haiku-negotiation configs'")
    print("    deltas: if local's delta is much smaller (or reversed), that's evidence")
    print("    the local model isn't using this context well, similar to its earlier")
    print("    struggles with population extraction and JSON formatting.")
    print("  - Configs sharing a negotiation model (see NOTE above) should land close to")
    print("    each other -- that's a sanity check on consistency, not two separate results.")
    print("  - This only tests whether lookahead_context changes behavior at all -- it")
    print("    doesn't test whether that behavior change improves real outcomes (that")
    print("    needs a full scenario run instead).")


if __name__ == "__main__":
    main()