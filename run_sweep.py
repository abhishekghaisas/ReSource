"""
Runs every config in MODEL_CONFIGS against one or more named disaster
scenarios and produces a comparison table: optimality gap vs. cost, broken
down by agent role.

THREE SEPARATE SOURCES OF NOISE/VARIATION, THREE SEPARATE KNOBS:

  --repeats N     Re-runs the SAME scenario N times to quantify MODEL-SAMPLING
                   noise (hosted-model calls are non-deterministic even at
                   temperature=0 -- haiku_everywhere's gap swung from -8.9%
                   to -25.7% between two runs of the identical scenario).

  --scenarios ... Runs across DIFFERENT disaster scenarios to quantify
                   SCENARIO-GENERALIZATION: does a config's ranking hold up
                   across qualitatively different crises, or was it just true
                   for one particular seed?

Each entry in --scenarios is "disaster_type:seed", e.g. "wildfire:7". The
three disaster types (flood, wildfire, earthquake) have deliberately
different damage dynamics (see simulation/world.py):
  - flood:      damage is gradual and uniformly random across the map.
  - wildfire:   damage spreads outward from an epicenter, worsening every
                timestep -- a predictable, escalating threat.
  - earthquake: damage is front-loaded (routes blocked and depot stock
                destroyed at t=0), then quiet aftershocks -- a sudden-onset
                shock rather than a slow build.

A config that wins on flood but loses on wildfire and earthquake is not a
validated general finding -- it's a result specific to gradual, uniform
damage. Testing across all three is what makes a "config X wins" claim
defensible rather than an artifact of one convenient scenario.

Sonnet-based configs (sonnet_everywhere, local_parsing_sonnet_negotiation)
are EXCLUDED from the default --configs list -- dropped from consideration
after forced extended-thinking collided with a fixed token budget, causing
truncation failures that inflated effective cost ~2x through wasted retries.
Still selectable via --configs if you want to revisit that.

Usage:
    # Smoke test (fast, not a real result -- one scenario, one run each)
    python run_sweep.py --timesteps 15 --scenarios flood:42

    # Real result: 3 disaster types x 5 repeats each = 15 runs per config
    python run_sweep.py --timesteps 15 --repeats 5 \
        --scenarios flood:42 wildfire:7 earthquake:99

    python run_sweep.py --timesteps 15 --repeats 5 \
        --scenarios flood:42 wildfire:7 earthquake:99 \
        --configs local_everywhere haiku_everywhere
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import statistics
import sys
import time

from optimizer.baseline import solve_optimal_allocation, unmet_urgent_need
from run_simulation import MODEL_CONFIGS, run_multi_agent_system
from simulation.world import DISASTER_TYPES, advance_disaster, make_scenario

DEFAULT_CONFIGS = [
    "local_everywhere",
    "haiku_everywhere",
    "local_parsing_haiku_negotiation",
]

DEFAULT_SCENARIOS = ["flood:42", "wildfire:7", "earthquake:99"]


def parse_scenario(spec: str):
    """'wildfire:7' -> ('wildfire', 7). Raises a clear error on bad input."""
    if ":" not in spec:
        raise argparse.ArgumentTypeError(
            f"scenario must be 'disaster_type:seed', got {spec!r}")
    disaster_type, seed_str = spec.split(":", 1)
    if disaster_type not in DISASTER_TYPES:
        raise argparse.ArgumentTypeError(
            f"disaster_type must be one of {DISASTER_TYPES}, got {disaster_type!r}")
    try:
        seed = int(seed_str)
    except ValueError:
        raise argparse.ArgumentTypeError(f"seed must be an integer, got {seed_str!r}")
    return (disaster_type, seed)


def scenario_label(disaster_type: str, seed: int) -> str:
    return f"{disaster_type}:{seed}"


def run_optimizer_once(disaster_type: str, seed: int, timesteps: int) -> float:
    world = make_scenario(seed=seed, disaster_type=disaster_type)
    rng = random.Random(seed)
    total_unmet = 0.0
    for _ in range(timesteps):
        advance_disaster(world, rng)
        alloc = solve_optimal_allocation(world)
        for (depot_id, shelter_id, resource), qty in alloc.items():
            world.depots[depot_id].stock[resource] = max(
                0.0, world.depots[depot_id].stock.get(resource, 0.0) - qty)
        total_unmet += unmet_urgent_need(world, alloc)
    return total_unmet


def run_one_trial(config_name: str, disaster_type: str, seed: int, timesteps: int,
                   api_key: str, optimal_unmet: float):
    world = make_scenario(seed=seed, disaster_type=disaster_type)
    rng = random.Random(seed)

    t0 = time.time()
    agent_unmet, tracker = run_multi_agent_system(world, rng, timesteps, config_name, api_key)
    wall_time = time.time() - t0

    gap_pct = ((agent_unmet - optimal_unmet) / optimal_unmet * 100
               if optimal_unmet > 0 else float("nan"))
    cost_by_role = tracker.cost_by_role()

    return {
        "unmet_need": agent_unmet,
        "optimality_gap_pct": gap_pct,
        "total_cost_usd": tracker.total_cost(),
        "wall_time_s": wall_time,
        "cost_field_report": cost_by_role.get("field_report", 0.0),
        "cost_depot_negotiation": cost_by_role.get("depot_negotiation", 0.0),
        "cost_dispatcher": cost_by_role.get("dispatcher", 0.0),
    }


def mean_std(values):
    values = [v for v in values if v is not None and v == v]  # drop None/NaN
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=15)
    parser.add_argument("--scenarios", type=parse_scenario, nargs="+", default=None,
                         metavar="TYPE:SEED",
                         help="One or more 'disaster_type:seed' scenarios, e.g. "
                              "'flood:42 wildfire:7 earthquake:99'. Defaults to "
                              f"one of each disaster type: {DEFAULT_SCENARIOS}.")
    parser.add_argument("--repeats", type=int, default=1,
                         help="Run each (config, scenario) pair this many times "
                              "to quantify model-sampling variance. Recommended "
                              ">=3 before trusting any cross-config comparison.")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS,
                         choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--out", default="sweep_results.csv",
                         help="Per (config, scenario) summary, mean+/-std across repeats.")
    parser.add_argument("--overall-out", default="sweep_results_overall.csv",
                         help="Per-config summary, mean+/-std pooled across ALL "
                              "scenarios and repeats -- the headline table.")
    parser.add_argument("--raw-out", default="sweep_results_raw.csv",
                         help="One row per individual (config, scenario, repeat) run.")
    args = parser.parse_args()
    scenarios = args.scenarios if args.scenarios else [parse_scenario(s) for s in DEFAULT_SCENARIOS]

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    uses_claude = any("claude" in "".join(MODEL_CONFIGS[c]) for c in args.configs)
    if uses_claude and not api_key:
        print("WARNING: one or more selected configs use Claude models but "
              "ANTHROPIC_API_KEY is not set. Those configs will fail. "
              "Run `export ANTHROPIC_API_KEY=sk-...` first, or restrict "
              "--configs to local_everywhere only.", file=sys.stderr)

    labels = [scenario_label(dt, s) for dt, s in scenarios]
    print(f"Scenarios: {labels}   Repeats per (config, scenario): {args.repeats}")
    print(f"Total runs per config: {len(scenarios) * args.repeats}\n")

    # Optimizer baseline depends on the scenario (disaster type + seed) -- compute once each.
    optimal_by_scenario = {}
    for disaster_type, seed in scenarios:
        label = scenario_label(disaster_type, seed)
        optimal_by_scenario[label] = run_optimizer_once(disaster_type, seed, args.timesteps)
        print(f"Optimizer baseline ({label}): unmet_need={optimal_by_scenario[label]:.2f}")
    print()

    raw_rows = []
    per_scenario_rows = []   # one row per (config, scenario): mean/std across repeats
    overall_rows = []        # one row per config: mean/std pooled across all scenarios+repeats

    for config_name in args.configs:
        print(f"=== {config_name} ===")
        all_trials_this_config = []  # pooled across every scenario, for the overall table

        for disaster_type, seed in scenarios:
            label = scenario_label(disaster_type, seed)
            optimal_unmet = optimal_by_scenario[label]
            trials = []
            for rep in range(args.repeats):
                try:
                    result = run_one_trial(config_name, disaster_type, seed, args.timesteps,
                                            api_key, optimal_unmet)
                except Exception as e:
                    print(f"  {label} repeat {rep+1}/{args.repeats} FAILED: {e}")
                    raw_rows.append({"config": config_name, "scenario": label, "repeat": rep,
                                      "status": "FAILED", "error": str(e)})
                    continue
                trials.append(result)
                all_trials_this_config.append(result)
                raw_rows.append({"config": config_name, "scenario": label, "repeat": rep,
                                  "status": "OK", "error": "",
                                  **{k: round(v, 6) for k, v in result.items()}})
                print(f"  {label} repeat {rep+1}/{args.repeats}: "
                      f"unmet_need={result['unmet_need']:.2f}  "
                      f"gap={result['optimality_gap_pct']:+.1f}%  "
                      f"cost=${result['total_cost_usd']:.6f}")

            if not trials:
                per_scenario_rows.append({"config": config_name, "scenario": label,
                                           "status": "ALL_FAILED", "n_ok": 0})
                continue

            gap_mean, gap_std = mean_std([t["optimality_gap_pct"] for t in trials])
            unmet_mean, unmet_std = mean_std([t["unmet_need"] for t in trials])
            cost_mean, cost_std = mean_std([t["total_cost_usd"] for t in trials])
            per_scenario_rows.append({
                "config": config_name, "scenario": label,
                "status": "OK" if len(trials) == args.repeats else "PARTIAL",
                "n_ok": len(trials),
                "optimal_unmet_need": round(optimal_unmet, 2),
                "unmet_need_mean": round(unmet_mean, 2),
                "unmet_need_std": round(unmet_std, 2),
                "optimality_gap_pct_mean": round(gap_mean, 1),
                "optimality_gap_pct_std": round(gap_std, 1),
                "total_cost_usd_mean": round(cost_mean, 6),
                "total_cost_usd_std": round(cost_std, 6),
            })
            print(f"  {label} SUMMARY: unmet_need={unmet_mean:.1f} (std={unmet_std:.1f}, "
                  f"optimizer={optimal_unmet:.1f})  gap={gap_mean:+.1f}% (std={gap_std:.1f})  "
                  f"cost=${cost_mean:.6f} (std={cost_std:.6f})  n={len(trials)}/{args.repeats}")

        # Pooled across every scenario for this config -- the headline number.
        if all_trials_this_config:
            gap_mean, gap_std = mean_std([t["optimality_gap_pct"] for t in all_trials_this_config])
            unmet_mean, unmet_std = mean_std([t["unmet_need"] for t in all_trials_this_config])
            cost_mean, cost_std = mean_std([t["total_cost_usd"] for t in all_trials_this_config])
            per_scenario_gaps = [r["optimality_gap_pct_mean"] for r in per_scenario_rows
                                   if r["config"] == config_name and "optimality_gap_pct_mean" in r]
            gap_range = (max(per_scenario_gaps) - min(per_scenario_gaps)) if len(per_scenario_gaps) > 1 else 0.0
            total_optimal_unmet = sum(optimal_by_scenario[scenario_label(dt, s)] for dt, s in scenarios)
            overall_rows.append({
                "config": config_name,
                "n_scenarios": len(scenarios),
                "n_total_runs_ok": len(all_trials_this_config),
                "n_total_runs_attempted": len(scenarios) * args.repeats,
                "total_optimal_unmet_need_across_scenarios": round(total_optimal_unmet, 2),
                "unmet_need_mean": round(unmet_mean, 2),
                "unmet_need_std": round(unmet_std, 2),
                "optimality_gap_pct_mean": round(gap_mean, 1),
                "optimality_gap_pct_std": round(gap_std, 1),
                "gap_range_across_scenario_means": round(gap_range, 1),
                "total_cost_usd_mean": round(cost_mean, 6),
                "total_cost_usd_std": round(cost_std, 6),
            })
            print(f"  POOLED across {len(scenarios)} scenario(s): "
                  f"unmet_need={unmet_mean:.1f} (std={unmet_std:.1f})  "
                  f"gap={gap_mean:+.1f}% (std={gap_std:.1f}, "
                  f"scenario-mean range={gap_range:.1f}pp)  "
                  f"cost=${cost_mean:.6f}\n")
        else:
            overall_rows.append({"config": config_name, "n_scenarios": len(scenarios),
                                  "status": "ALL_FAILED"})
            print()

    if raw_rows:
        raw_fields = sorted(set().union(*[r.keys() for r in raw_rows]))
        with open(args.raw_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=raw_fields)
            writer.writeheader()
            writer.writerows(raw_rows)

    if per_scenario_rows:
        per_scenario_fields = sorted(set().union(*[r.keys() for r in per_scenario_rows]))
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=per_scenario_fields)
            writer.writeheader()
            writer.writerows(per_scenario_rows)

    if overall_rows:
        overall_fields = sorted(set().union(*[r.keys() for r in overall_rows]))
        with open(args.overall_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=overall_fields)
            writer.writeheader()
            writer.writerows(overall_rows)

    print("=" * 110)
    print(f"OVERALL SUMMARY  (pooled across scenarios={labels}, {args.repeats} repeat(s)/scenario)")
    print("=" * 110)
    header = (f"{'config':<38} {'unmet_need (mean±std)':>24} {'gap% (mean±std)':>20} "
               f"{'scenario-range(pp)':>18} {'cost($, mean)':>16} {'n_ok'}")
    print(header)
    print("-" * len(header))
    for row in overall_rows:
        if "optimality_gap_pct_mean" in row:
            print(f"{row['config']:<38} "
                  f"{row['unmet_need_mean']:>10.1f} ± {row['unmet_need_std']:<9.1f} "
                  f"{row['optimality_gap_pct_mean']:>+7.1f}% ± {row['optimality_gap_pct_std']:<8.1f} "
                  f"{row['gap_range_across_scenario_means']:>16.1f}   "
                  f"{row['total_cost_usd_mean']:>14.6f} "
                  f"{row['n_total_runs_ok']}/{row['n_total_runs_attempted']}")
        else:
            print(f"{row['config']:<38} {'ALL FAILED':>20}")
    print()
    print("NOTE: 'gap%' divides by the optimizer's own unmet need for that scenario --")
    print("if that denominator is small (e.g. a scenario with generous resupply), a tiny")
    print("absolute difference can look like a huge percentage. Always sanity-check gap%")
    print("against the absolute 'unmet_need' column before trusting it.")
    print()
    print(f"Per-(config,scenario) detail:  {args.out}")
    print(f"Pooled headline table:         {args.overall_out}")
    print(f"Every individual run:          {args.raw_out}")

    scenario_types_used = set(dt for dt, _ in scenarios)
    if len(scenario_types_used) < 3:
        missing = set(DISASTER_TYPES) - scenario_types_used
        print(f"\nNOTE: only tested disaster type(s) {sorted(scenario_types_used)}. "
              f"Results say nothing about generalization to {sorted(missing)} -- "
              "add scenarios of those types before treating a ranking as validated "
              "across disaster types.")
    else:
        print(f"\nCheck 'gap_range_across_scenario_means' in {args.overall_out}: a large")
        print("range means a config's ranking is disaster-type-dependent, not a stable")
        print("property of the config -- worth reporting explicitly either way.")

    if args.repeats == 1:
        print("\nNOTE: --repeats=1 means these numbers include unquantified model-sampling")
        print("noise on top of scenario variance. Re-run with --repeats 3+ for a fully")
        print("defensible result.")


if __name__ == "__main__":
    main()