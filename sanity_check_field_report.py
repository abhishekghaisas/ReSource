"""
Sanity check: is the local model's Field Report Agent actually extracting
meaningful urgency signal from noisy situation reports, or is it mostly
falling back to the neutral {0.5, 0.5, 0.5} guess?

Run this locally (requires `ollama serve` running, per your setup):

    python sanity_check_field_report.py --timesteps 15 --seed 42

Read the output as: if "parsed" tracks "true" reasonably well (both trend
up/down together, similar ordering across shelters), the cheap tier is
doing real work. If "parsed" is mostly 0.5/0.5/0.5 regardless of "true",
the local model is failing to follow the JSON-extraction instruction
reliably and you're silently losing signal -- worth knowing before you
trust the local_everywhere cost numbers.
"""

from __future__ import annotations

import argparse
import random

from agents.base_agent import CostTracker, LLMBackend
from agents.field_report_agent import FieldReportAgent
from simulation.world import RESOURCE_TYPES, advance_disaster, make_scenario


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    world = make_scenario(seed=args.seed)
    rng = random.Random(args.seed)
    tracker = CostTracker()
    agent = FieldReportAgent(LLMBackend("local", tracker, "field_report_sanity_check"))

    fallback_count = 0
    total_count = 0
    abs_errors = []
    signed_errors = []  # parsed - true; positive = over-estimate, negative = under-estimate

    print(f"{'t':>3} {'shelter':>8} {'resource':>9} {'true':>6} {'parsed':>7}  report text")
    print("-" * 100)

    for t in range(args.timesteps):
        advance_disaster(world, rng)
        for shelter in world.shelters.values():
            report_text = shelter.situation_report(rng)
            parsed = agent.parse(report_text, t)

            for resource in RESOURCE_TYPES:
                true_urgency = round(shelter.urgency(resource), 2)
                parsed_urgency = round(parsed[resource], 2)
                total_count += 1
                if parsed == {"food": 0.5, "water": 0.5, "medical": 0.5}:
                    fallback_count += 1
                abs_errors.append(abs(true_urgency - parsed_urgency))
                signed_errors.append(parsed_urgency - true_urgency)

                # Only print a flagged row when the gap is large, to keep
                # output readable -- change this to print every row if you
                # want the full trace. Full (untruncated) report text so
                # you can actually see all three resource sentences.
                if abs(true_urgency - parsed_urgency) > 0.35:
                    print(f"{t:>3} {shelter.id:>8} {resource:>9} {true_urgency:>6} "
                          f"{parsed_urgency:>7}  {report_text}")

    over_estimates = [e for e in signed_errors if e > 0.15]
    under_estimates = [e for e in signed_errors if e < -0.15]
    accurate = [e for e in signed_errors if -0.15 <= e <= 0.15]

    print()
    print("=" * 60)
    print(f"Total (shelter, resource, timestep) observations: {total_count}")
    print(f"Fell back to neutral 0.5/0.5/0.5 guess: {fallback_count} times "
          f"({100*fallback_count/total_count:.1f}% of shelter-reports)")
    print(f"Mean absolute error (true vs. parsed urgency): "
          f"{sum(abs_errors)/len(abs_errors):.3f}")
    print(f"Mean signed error (parsed - true): "
          f"{sum(signed_errors)/len(signed_errors):+.3f}  "
          f"(positive = systematic over-estimation)")
    print()
    print("Directional breakdown (threshold 0.15):")
    print(f"  Over-estimated (parsed > true):  {len(over_estimates):>4} / {total_count} "
          f"({100*len(over_estimates)/total_count:.1f}%)  mean magnitude "
          f"{sum(over_estimates)/len(over_estimates):.2f}" if over_estimates else "  Over-estimated: 0")
    print(f"  Under-estimated (parsed < true): {len(under_estimates):>4} / {total_count} "
          f"({100*len(under_estimates)/total_count:.1f}%)  mean magnitude "
          f"{sum(under_estimates)/len(under_estimates):.2f}" if under_estimates else "  Under-estimated: 0")
    print(f"  Accurate (within 0.15):          {len(accurate):>4} / {total_count} "
          f"({100*len(accurate)/total_count:.1f}%)")
    print()
    print("Interpretation:")
    print("  - Fallback rate > ~20%: the local model is frequently failing to")
    print("    produce valid JSON -- tighten the prompt or add a retry.")
    print("  - Mean absolute error > ~0.3: even when JSON parses, the model is")
    print("    reading the noisy text poorly -- consider a slightly larger local")
    print("    model, or note this as a real limitation of the cheap tier.")
    print(tracker.summary())


if __name__ == "__main__":
    main()