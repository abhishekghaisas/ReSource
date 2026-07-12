"""
Directly tests InterDepotCoordinatorAgent's decision-making in isolation,
same philosophy as run_lookahead_test.py for Phase D -- controlled inputs,
not inferred from a full noisy scenario run.

Two cases:

  Case A ("clear surplus/deficit -- SHOULD transfer"): D1 has abundant
  stock and low urgency among its reachable shelters (a real surplus it
  isn't using). D2 has scarce stock and high urgency, several critical
  shelters (a real deficit). A sensible coordinator should recommend a
  transfer from D1 to D2.

  Case B ("balanced -- should NOT transfer"): both depots have similar
  stock and similar, moderate urgency. Neither has a clear surplus or
  deficit. A sensible coordinator should recommend NO transfer -- an
  agent that transfers eagerly even without a clear case wastes a vehicle
  just as badly as one that never transfers at all when it should.

This tests both directions of the capability, not just "can it ever
transfer" -- restraint when unwarranted matters as much as action when
warranted.

Usage:
    python run_interdepot_test.py --repeats 5
    python run_interdepot_test.py --configs haiku_everywhere --repeats 10
"""

from __future__ import annotations

import argparse
import os

from agents.base_agent import CostTracker, LLMBackend
from agents.inter_depot_agent import InterDepotCoordinatorAgent
from run_simulation import MODEL_CONFIGS

AVAILABLE_VEHICLES = {
    "T1": {"capacity": 30.0, "speed": 2},
    "T2": {"capacity": 25.0, "speed": 1},
}

CASE_A_SHOULD_TRANSFER = {
    "D1": {"stock": {"food": 100.0, "water": 100.0, "medical": 100.0},
           "avg_reachable_urgency": 0.1, "n_critical_reachable_shelters": 0},
    "D2": {"stock": {"food": 5.0, "water": 5.0, "medical": 5.0},
           "avg_reachable_urgency": 0.9, "n_critical_reachable_shelters": 3},
}

CASE_B_SHOULD_NOT_TRANSFER = {
    "D1": {"stock": {"food": 50.0, "water": 50.0, "medical": 50.0},
           "avg_reachable_urgency": 0.5, "n_critical_reachable_shelters": 1},
    "D2": {"stock": {"food": 50.0, "water": 50.0, "medical": 50.0},
           "avg_reachable_urgency": 0.5, "n_critical_reachable_shelters": 1},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+",
                         default=["local_everywhere", "haiku_everywhere",
                                  "local_parsing_haiku_negotiation_dispatch"],
                         choices=list(MODEL_CONFIGS))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--debug", action="store_true",
                         help="Print raw model output for every call, to distinguish a "
                              "genuine 'no transfer' judgment from a silent parsing failure "
                              "that defaults to None regardless of input.")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if args.debug:
        original_call = LLMBackend.call

        def debug_call(self, system_prompt, user_prompt, timestep, max_tokens=300):
            text = original_call(self, system_prompt, user_prompt, timestep, max_tokens)
            preview = text.strip().replace("\n", " ")[:300]
            print(f"    [DEBUG model={self.model}] raw_response={preview!r}")
            return text

        LLMBackend.call = debug_call


    model_to_configs: dict = {}
    for c in args.configs:
        depot_model = MODEL_CONFIGS[c][1]
        model_to_configs.setdefault(depot_model, []).append(c)
    for model, configs in model_to_configs.items():
        if len(configs) > 1:
            print(f"NOTE: {', '.join(configs)} all use '{model}' for coordination -- "
                  f"expect near-identical results between them here.")
    print()

    all_summaries = {}

    for config_name in args.configs:
        depot_model = MODEL_CONFIGS[config_name][1]
        print(f"=== {config_name}  (coordination model: {depot_model}) ===")

        case_a_transferred = 0
        case_a_correct_direction = 0
        case_b_transferred = 0

        for rep in range(args.repeats):
            tracker = CostTracker()
            agent = InterDepotCoordinatorAgent(
                LLMBackend(depot_model, tracker, "inter_depot_coordination", api_key))

            decision_a = agent.coordinate(CASE_A_SHOULD_TRANSFER, AVAILABLE_VEHICLES, timestep=0)
            decision_b = agent.coordinate(CASE_B_SHOULD_NOT_TRANSFER, AVAILABLE_VEHICLES, timestep=0)

            a_transferred = decision_a is not None
            a_correct_dir = (a_transferred and decision_a["source_depot_id"] == "D1"
                              and decision_a["dest_depot_id"] == "D2")
            b_transferred = decision_b is not None

            case_a_transferred += int(a_transferred)
            case_a_correct_direction += int(a_correct_dir)
            case_b_transferred += int(b_transferred)

            print(f"  repeat {rep+1}/{args.repeats}: "
                  f"Case A={'TRANSFER ' + decision_a['source_depot_id'] + '->' + decision_a['dest_depot_id'] if a_transferred else 'no transfer'}   "
                  f"Case B={'TRANSFER (unwarranted!)' if b_transferred else 'no transfer (correct)'}")

        print(f"  SUMMARY: Case A transferred {case_a_transferred}/{args.repeats} "
              f"({case_a_correct_direction}/{args.repeats} correct direction D1->D2)   "
              f"Case B transferred {case_b_transferred}/{args.repeats} (should be 0)")
        print()

        all_summaries[config_name] = {
            "case_a_transfer_rate": case_a_transferred / args.repeats,
            "case_a_correct_direction_rate": case_a_correct_direction / args.repeats,
            "case_b_transfer_rate": case_b_transferred / args.repeats,
        }

    print("=" * 90)
    print("CROSS-CONFIG SUMMARY")
    print("=" * 90)
    print(f"{'config':<42} {'Case A: transfer rate':>22} {'(correct dir)':>14} "
          f"{'Case B: transfer rate':>22}")
    print("-" * 90)
    for config_name, s in all_summaries.items():
        print(f"{config_name:<42} {s['case_a_transfer_rate']:>21.0%}  "
              f"{s['case_a_correct_direction_rate']:>13.0%} "
              f"{s['case_b_transfer_rate']:>22.0%}")
    print()
    print("How to read this:")
    print("  - Case A transfer rate should be HIGH (ideally 100%) and correct-direction rate")
    print("    should match it -- a real surplus/deficit case should reliably trigger a")
    print("    correctly-directed transfer, not a coin flip.")
    print("  - Case B transfer rate should be LOW (ideally 0%) -- an agent that transfers")
    print("    even in a balanced situation is wasting vehicles on unwarranted moves, which")
    print("    is its own failure mode, not a safe default.")
    print("  - Compare local_everywhere against the Haiku-coordination configs: if local's")
    print("    Case A rate is much lower, or its Case B rate is much higher, that's evidence")
    print("    the local model isn't reasoning about this well -- consistent with its")
    print("    struggles on other structured-judgment tasks earlier in this project.")


if __name__ == "__main__":
    main()