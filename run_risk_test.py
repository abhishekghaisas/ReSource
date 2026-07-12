"""
Directly tests DepotAgent's risk-taking judgment (Phase R) in isolation,
same philosophy as run_lookahead_test.py and run_interdepot_test.py --
controlled inputs, not inferred from a full noisy scenario run.

Two shelters, S1 (risky route, risk=0.6) and S2 (safe route, risk=0,
moderate urgency, held constant across both cases so it's a genuine
alternative use for the same stock, not a strawman with nothing else to do):

  Case A ("worth the risk"): S1 is critical (urgency 0.9, real population).
  A sensible negotiator should still commit stock to S1 despite the risk --
  the need justifies the gamble.

  Case B ("not worth the risk"): S1 is low-urgency (0.15), same risky route,
  same population. A sensible negotiator should prefer the safe S2 and NOT
  throw a vehicle at real danger for a shelter that isn't actually critical.

This tests whether risk changes behavior in the expected DIRECTION relative
to urgency -- not whether any single allocation number is "correct" (both
cases have reasonable answers; what matters is S1's allocation differing
sensibly between them).

Usage:
    python run_risk_test.py --repeats 5
    python run_risk_test.py --configs haiku_everywhere --repeats 10
"""

from __future__ import annotations

import argparse
import os

from agents.base_agent import CostTracker, LLMBackend
from agents.depot_agent import DepotAgent
from run_simulation import MODEL_CONFIGS

STOCK = {"food": 30.0, "water": 30.0, "medical": 30.0}
TRANSPORT_CAPACITY = 100.0
ROUTE_STATUS = {"S1": True, "S2": True}  # both reachable -- S1 is risky, not hard-blocked
ROUTE_RISK = {"S1": 0.6}  # S2 absent -- no meaningful risk on that route

CASE_A_WORTH_THE_RISK = {
    "S1": {"food": 0.9, "water": 0.9, "medical": 0.9, "population_estimate": 300},
    "S2": {"food": 0.5, "water": 0.5, "medical": 0.5, "population_estimate": 300},
}
CASE_B_NOT_WORTH_THE_RISK = {
    "S1": {"food": 0.15, "water": 0.15, "medical": 0.15, "population_estimate": 300},
    "S2": {"food": 0.5, "water": 0.5, "medical": 0.5, "population_estimate": 300},
}


def s1_allocated(alloc: dict) -> float:
    return sum(alloc.get("S1", {}).values())


def s2_allocated(alloc: dict) -> float:
    return sum(alloc.get("S2", {}).values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+",
                         default=["local_everywhere", "haiku_everywhere",
                                  "local_parsing_haiku_negotiation_dispatch"],
                         choices=list(MODEL_CONFIGS))
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")

    model_to_configs: dict = {}
    for c in args.configs:
        depot_model = MODEL_CONFIGS[c][1]
        model_to_configs.setdefault(depot_model, []).append(c)
    for model, configs in model_to_configs.items():
        if len(configs) > 1:
            print(f"NOTE: {', '.join(configs)} all use '{model}' for negotiation -- "
                  f"expect near-identical results between them here.")
    print()

    all_summaries = {}

    for config_name in args.configs:
        depot_model = MODEL_CONFIGS[config_name][1]
        print(f"=== {config_name}  (negotiation model: {depot_model}) ===")

        s1_a_vals, s1_b_vals = [], []
        s2_a_vals, s2_b_vals = [], []

        for rep in range(args.repeats):
            tracker = CostTracker()
            depot_agent = DepotAgent(LLMBackend(depot_model, tracker, "depot_negotiation", api_key))

            alloc_a = depot_agent.allocate(
                "D1", dict(STOCK), TRANSPORT_CAPACITY, CASE_A_WORTH_THE_RISK, timestep=0,
                route_status=ROUTE_STATUS, route_risk=ROUTE_RISK)
            s1_a, s2_a = s1_allocated(alloc_a), s2_allocated(alloc_a)
            s1_a_vals.append(s1_a)
            s2_a_vals.append(s2_a)

            alloc_b = depot_agent.allocate(
                "D1", dict(STOCK), TRANSPORT_CAPACITY, CASE_B_NOT_WORTH_THE_RISK, timestep=0,
                route_status=ROUTE_STATUS, route_risk=ROUTE_RISK)
            s1_b, s2_b = s1_allocated(alloc_b), s2_allocated(alloc_b)
            s1_b_vals.append(s1_b)
            s2_b_vals.append(s2_b)

            print(f"  repeat {rep+1}/{args.repeats}: "
                  f"Case A (S1 critical) -> S1={s1_a:.1f} S2={s2_a:.1f}   "
                  f"Case B (S1 low urgency) -> S1={s1_b:.1f} S2={s2_b:.1f}")

        mean_s1_a = sum(s1_a_vals) / len(s1_a_vals)
        mean_s1_b = sum(s1_b_vals) / len(s1_b_vals)
        mean_s2_a = sum(s2_a_vals) / len(s2_a_vals)
        mean_s2_b = sum(s2_b_vals) / len(s2_b_vals)
        all_summaries[config_name] = (mean_s1_a, mean_s1_b, mean_s2_a, mean_s2_b)
        print(f"  SUMMARY: S1 mean  Case A={mean_s1_a:.2f}  Case B={mean_s1_b:.2f}  "
              f"delta={mean_s1_a - mean_s1_b:+.2f}")
        print(f"           S2 mean  Case A={mean_s2_a:.2f}  Case B={mean_s2_b:.2f}\n")

    print("=" * 92)
    print("CROSS-CONFIG SUMMARY")
    print("=" * 92)
    print(f"{'config':<42} {'S1 Case A':>10} {'S1 Case B':>10} {'S1 delta':>10} {'S2 Case A':>10} {'S2 Case B':>10}")
    print("-" * 92)
    for config_name, (a1, b1, a2, b2) in all_summaries.items():
        print(f"{config_name:<42} {a1:>10.2f} {b1:>10.2f} {a1-b1:>+10.2f} {a2:>10.2f} {b2:>10.2f}")
    print()
    print("How to read this:")
    print("  - S1 delta should be strongly POSITIVE: allocates much more to the risky shelter")
    print("    when it's genuinely critical (Case A) than when it isn't (Case B). Near-zero or")
    print("    negative delta means risk isn't being weighed against urgency at all.")
    print("  - S2 (the safe shelter) should get a reasonable share in Case B especially --")
    print("    a good negotiator prefers the safe option when the risky one isn't justified,")
    print("    rather than just declining to allocate anything at all.")
    print("  - Compare local_everywhere's delta against the Haiku-negotiation configs': if")
    print("    local's delta is much smaller or inconsistent, that's evidence it isn't")
    print("    reasoning about the urgency-vs-risk tradeoff the way Haiku does.")


if __name__ == "__main__":
    main()