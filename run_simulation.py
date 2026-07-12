"""
Main entry point: runs a scenario through the decentralized multi-agent
system AND the centralized optimizer baseline on identical world states,
then reports the optimality gap and the cost of getting there.

Usage:
    python run_simulation.py --config haiku_everywhere
    python run_simulation.py --config local_parsing_sonnet_negotiation
    python run_simulation.py --config sonnet_everywhere

Requires ANTHROPIC_API_KEY in the environment for any config using Claude,
and a running Ollama server (ollama serve) for any config using "local".
"""

from __future__ import annotations

import argparse
import copy
import os
import random

from agents.base_agent import CostTracker, LLMBackend
from agents.depot_agent import DepotAgent
from agents.dispatcher_agent import DispatcherAgent
from agents.field_report_agent import FieldReportAgent
from agents.inter_depot_agent import InterDepotCoordinatorAgent
from optimizer.baseline import solve_optimal_allocation, unmet_urgent_need
from simulation.world import (RESOURCE_TYPES, advance_disaster, dispatch_shipment,
                               dispatch_transfer, make_scenario, resolve_arrivals,
                               resolve_risk_outcome, resolve_transfer_arrivals,
                               route_risk_score, timesteps_until_next_resupply)

URGENCY_HISTORY_LENGTH = 3  # how many past rounds' average urgency DepotAgent gets to see


def average_urgency(shelter_urgency_estimates: dict) -> float:
    """
    Mean urgency across every shelter and resource this round (excludes
    non-resource keys like population_estimate). Used to build the rolling
    trend history in lookahead_context (Phase D) -- shared across all
    orchestration scripts rather than duplicated, since every one of them
    needs the identical computation to build a comparable trend signal.
    """
    values = [v for est in shelter_urgency_estimates.values()
              for k, v in est.items() if k in RESOURCE_TYPES]
    return sum(values) / len(values) if values else 0.0


CRITICAL_URGENCY_THRESHOLD = 0.5  # matches optimizer.baseline.unmet_urgent_need's default


def build_depot_summaries(world, shelter_urgency_estimates: dict) -> dict:
    """
    Per-depot summary for InterDepotCoordinatorAgent (Phase C): current
    stock, plus average urgency and critical-shelter count among ONLY the
    shelters that depot can currently reach (route_status-aware, same
    reachability check DepotAgent and DispatcherAgent already use). Shared
    across orchestration scripts for the same reason average_urgency() is.
    """
    summaries = {}
    for depot in world.depots.values():
        reachable_urgencies = []
        n_critical = 0
        for shelter in world.shelters.values():
            if world.route_damaged(depot.position, shelter.position):
                continue
            est = shelter_urgency_estimates.get(shelter.id, {})
            vals = [v for k, v in est.items() if k in RESOURCE_TYPES]
            if not vals:
                continue
            avg = sum(vals) / len(vals)
            reachable_urgencies.append(avg)
            if avg >= CRITICAL_URGENCY_THRESHOLD:
                n_critical += 1
        summaries[depot.id] = {
            "stock": dict(depot.stock),
            "avg_reachable_urgency": round(
                sum(reachable_urgencies) / len(reachable_urgencies), 3) if reachable_urgencies else 0.0,
            "n_critical_reachable_shelters": n_critical,
        }
        # Phase R: risk score to every OTHER depot, for InterDepotCoordinatorAgent.
        # Only included when nonzero, same convention as route_risk elsewhere,
        # to keep the payload small on the (common) case of no elevated risk.
        transfer_risk = {}
        for other in world.depots.values():
            if other.id == depot.id:
                continue
            r = route_risk_score(world, depot.position, other.position)
            if r > 0:
                transfer_risk[other.id] = round(r, 3)
        if transfer_risk:
            summaries[depot.id]["transfer_risk"] = transfer_risk
    return summaries


def rotate_depots_for_fairness(world, t: int) -> list:
    """
    Returns depots ordered by ASCENDING cumulative vehicle claims so far
    (world.depot_vehicle_claims) -- whichever depot has been served LEAST
    goes first, a genuine fair-share mechanism rather than simple turn
    alternation. Confirmed bug and confirmed fix history (see FINDINGS.md
    "Depot Processing Order Fleet-Starvation Bug"):

    1. A FIXED iteration order (world.depots.values() always yields the
       same depot first) let one depot see 0 available vehicles on
       literally every day of a real run -- confirmed via direct trace.
    2. A first-pass fix using simple day-parity alternation (depot order
       flips every timestep) was ALSO confirmed insufficient: it can
       resonate with a periodic vehicle round-trip cycle (e.g. a 2-day
       alternation lining up with a ~2-day round trip means the
       under-served depot's "first" days always land when the fleet is
       still mid-transit from the over-served depot's immediately
       preceding turn) and fail to actually redistribute access.
    3. This version fixes that by tracking real cumulative outcomes
       instead of a fixed schedule: the depot that's actually gotten
       fewer vehicles so far always goes first, which can't resonate with
       any periodic cycle because it directly responds to the actual
       history rather than following a predetermined pattern.

    Ties (including every depot at 0 claims, e.g. at the very start of a
    run) fall back to original dict order for determinism.
    """
    depot_list = list(world.depots.values())
    if len(depot_list) <= 1:
        return depot_list
    return sorted(depot_list, key=lambda d: world.depot_vehicle_claims.get(d.id, 0))


def compute_route_risk_for_depot(world, depot) -> dict:
    """
    {shelter_id: risk_score} for every shelter REACHABLE from this depot
    (route_damaged already False) that carries nonzero Phase R risk. Shared
    across orchestration scripts, same reasoning as average_urgency() and
    build_depot_summaries() above -- every script needs the identical
    computation to feed DepotAgent/DispatcherAgent consistently.
    """
    risk = {}
    for shelter in world.shelters.values():
        if world.route_damaged(depot.position, shelter.position):
            continue
        r = route_risk_score(world, depot.position, shelter.position)
        if r > 0:
            risk[shelter.id] = r
    return risk


def apply_shipment_with_risk(world, risk_rng, depot, shelter_id: str, resource: str, qty: float,
                              vehicle_id: str, t: int, risk_score: float) -> str:
    """
    Deducts the FULL requested quantity from depot stock (it physically
    left the depot on the truck regardless of what happens next), rolls a
    Phase R risk outcome if risk_score > 0, and dispatches a real Shipment
    for only the SURVIVING quantity (0 if totally lost, meaning no Shipment
    is created at all). Marks the vehicle destroyed on a total-loss
    outcome. Shared across orchestration scripts -- every one of them needs
    this identical sequence wherever a delivery is actually dispatched.

    CRITICAL: risk_rng MUST be a SEPARATE random.Random instance from the
    one driving advance_disaster()/situation reports/etc. -- NEVER the same
    object. Confirmed bug (see FINDINGS.md "RNG Stream Contamination"):
    passing the same shared rng here means the NUMBER of risky-shipment
    attempts (which varies by config -- e.g. haiku_everywhere ships far
    more often than local_everywhere) changes how many random draws get
    consumed before the next world-state draw (population growth, etc.),
    silently desyncing "ground truth" world state between configs that are
    supposed to be facing an IDENTICAL scenario under the same seed. A
    separate risk_rng means different risk-attempt counts only affect risk
    OUTCOMES, never the underlying world both configs are meant to share.

    Returns the outcome string ("success"/"partial_loss"/"total_loss") for
    logging/diagnostics.
    """
    depot.stock[resource] = max(0.0, depot.stock.get(resource, 0.0) - qty)
    outcome, surviving_fraction = resolve_risk_outcome(risk_rng, risk_score)
    surviving_qty = qty * surviving_fraction
    if surviving_qty > 0:
        dispatch_shipment(world, depot.id, shelter_id, resource, surviving_qty, vehicle_id, t)
    if outcome == "total_loss":
        world.transports[vehicle_id].destroyed = True
    return outcome


def apply_transfer_with_risk(world, risk_rng, source_depot, dest_depot_id: str, resource: str,
                              qty: float, vehicle_id: str, t: int, risk_score: float) -> str:
    """
    Phase R equivalent of apply_shipment_with_risk() for inter-depot
    transfers (Phase C) instead of shelter deliveries. Same reasoning:
    full quantity leaves the source depot regardless of outcome, only the
    surviving fraction actually gets a real Transfer dispatched.

    CRITICAL: risk_rng must be the same SEPARATE, dedicated RNG instance
    used for apply_shipment_with_risk -- see that function's docstring for
    why sharing the world's main rng here is a confirmed bug, not a
    simplification.
    """
    source_depot.stock[resource] = max(0.0, source_depot.stock.get(resource, 0.0) - qty)
    outcome, surviving_fraction = resolve_risk_outcome(risk_rng, risk_score)
    surviving_qty = qty * surviving_fraction
    if surviving_qty > 0:
        dispatch_transfer(world, source_depot.id, dest_depot_id, resource, surviving_qty, vehicle_id, t)
    if outcome == "total_loss":
        world.transports[vehicle_id].destroyed = True
    return outcome


MODEL_CONFIGS = {
    # name -> (field_report_model, depot_model, dispatcher_model)
    #
    # ARCHITECTURAL PRINCIPLE (established across three separate pieces of
    # evidence -- see FINDINGS.md "Recommended architecture" for the full
    # writeup): local models are reliable for EXTRACTION (field-report
    # parsing: pulling urgency/population out of noisy text, a single-entity
    # mechanical task) but not for REASONING tasks that require weighing
    # multiple entities against each other and making a judgment call under
    # tradeoffs -- negotiation, dispatch, and inter-depot coordination all
    # showed this independently:
    #   1. Negotiation: local DepotAgent measurably worse quality than Haiku.
    #   2. Dispatch: local DispatcherAgent discarded >85% of whatever
    #      negotiation-quality advantage existed upstream (dispatch-
    #      bottleneck diagnostic).
    #   3. Inter-depot coordination: local produced the IDENTICAL transfer
    #      recommendation regardless of whether the input showed a clear
    #      surplus/deficit or a perfectly balanced scenario -- not reasoning
    #      about the decision content at all, not just reasoning to a worse
    #      conclusion.
    # Recommended going forward: local ONLY for field_report_model; every
    # reasoning role (depot_model, dispatcher_model, and inter-depot
    # coordination, which reuses depot_model -- see agents/inter_depot_agent.py)
    # should be Haiku or stronger, regardless of cost-tier config name.
    "local_everywhere": ("local", "local", "local"),
    # Retained for historical/diagnostic comparison ONLY -- this is the
    # baseline every quality finding above was measured against, not a
    # recommended deployment choice. Local negotiation, dispatch, AND
    # coordination are all confirmed non-functional or poor quality (see
    # principle above); this config exists to quantify how much each
    # reasoning role's cost matters, not because it's a good option.
    "haiku_everywhere": ("claude-haiku-4-5-20251001", "claude-haiku-4-5-20251001", "claude-haiku-4-5-20251001"),
    # Retained for historical continuity (this was the original mixed-tier
    # cost-efficiency config, predating the dispatch-bottleneck diagnostic).
    # Leaves dispatch on local, which is now KNOWN to be a poor choice --
    # superseded by local_parsing_haiku_negotiation_dispatch below. Not
    # recommended for new work.
    "local_parsing_haiku_negotiation": ("local", "claude-haiku-4-5-20251001", "local"),
    # RECOMMENDED CONFIG: local only for field-report parsing (extraction),
    # Haiku for negotiation and dispatch -- and, since InterDepotCoordinatorAgent
    # reuses depot_model, Haiku for inter-depot coordination too. This is the
    # config that actually follows the architectural principle above; added
    # originally after the Phase A dispatch-bottleneck diagnostic (local
    # dispatch discarded >85% of the negotiation quality difference between
    # local and Haiku negotiation -- only ~14-15% of what either config's
    # DepotAgent decided actually made it onto a vehicle, with low vehicle
    # utilization pointing at decision quality, not capacity, as the
    # bottleneck), and further validated by the inter-depot coordination
    # test above.
    "local_parsing_haiku_negotiation_dispatch": (
        "local", "claude-haiku-4-5-20251001", "claude-haiku-4-5-20251001"),
}


def run_multi_agent_system(world, rng, n_timesteps: int, model_config: str,
                            anthropic_api_key: str):
    fr_model, depot_model, dispatch_model = MODEL_CONFIGS[model_config]
    tracker = CostTracker()

    # RNG STREAM SEPARATION (see FINDINGS.md "RNG Stream Contamination"):
    # risk_rng is a SEPARATE, independent stream from `rng`, derived once
    # HERE -- before any config-dependent behavior can diverge -- so that
    # every config draws this seed identically. From this point on, `rng`
    # drives world-ground-truth randomness only (population growth,
    # disaster progression, situation-report noise) and risk_rng drives
    # ONLY Phase R risk-outcome rolls. Without this separation, configs
    # that attempt different NUMBERS of risky shipments (e.g. haiku_
    # everywhere ships far more often than local_everywhere) consume a
    # different number of draws from a shared stream, silently desyncing
    # "identical" ground truth between configs that are supposed to face
    # the exact same scenario under the same seed.
    risk_rng = random.Random(rng.random())

    field_report_agent = FieldReportAgent(
        LLMBackend(fr_model, tracker, "field_report", anthropic_api_key))
    depot_agent = DepotAgent(
        LLMBackend(depot_model, tracker, "depot_negotiation", anthropic_api_key))
    dispatcher_agent = DispatcherAgent(
        LLMBackend(dispatch_model, tracker, "dispatcher", anthropic_api_key))
    # Phase C: reuses depot_model rather than being its own tunable role --
    # see agents/inter_depot_agent.py's module docstring for why.
    inter_depot_agent = InterDepotCoordinatorAgent(
        LLMBackend(depot_model, tracker, "inter_depot_coordination", anthropic_api_key))

    total_unmet = 0.0
    urgency_history = []  # rolling window, most recent last -- see Phase D

    for t in range(n_timesteps):
        advance_disaster(world, rng)

        # 0. Phase A: resolve any shipments dispatched in earlier rounds
        #    whose travel time has now elapsed. This -- not what gets
        #    decided later this same timestep -- is what actually counts
        #    as "delivered" for evaluating unmet need below. A delivery
        #    dispatched THIS round won't show up here until a future t.
        arrived_this_step = resolve_arrivals(world, t)
        # Phase C: resolve any inter-depot transfers that have arrived --
        # these add directly to the destination depot's stock, informational
        # return value not needed for the unmet-need calculation.
        resolve_transfer_arrivals(world, t)

        # 1. Every shelter files a noisy report; Field Report Agent parses it.
        shelter_urgency_estimates = {}
        for shelter in world.shelters.values():
            report_text = shelter.situation_report(rng)
            shelter_urgency_estimates[shelter.id] = field_report_agent.parse(report_text, t)

        # 1.5. Phase C: once per timestep (not per depot), decide whether any
        #      depot should transfer surplus stock to another before the
        #      regular per-depot negotiation/dispatch below. This runs FIRST
        #      so a transfer's vehicle usage is reflected in the transport
        #      capacity/availability the per-depot loop sees this round --
        #      a transfer genuinely competes with shelter deliveries for the
        #      same shared fleet, not a free side-channel.
        available_vehicles_for_coord = {
            tr.id: {"capacity": tr.capacity, "speed": tr.speed}
            for tr in world.transports.values()
            # Phase R: a destroyed vehicle never comes back.
            if tr.busy_until <= t and not tr.destroyed
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

        # 2. Each depot negotiates a desired allocation, then the Dispatcher
        #    decides what actually ships this round given real vehicle
        #    availability/capacity -- these are no longer the same thing.
        #    Only the SHIPPED portion leaves depot.stock; whatever DepotAgent
        #    wanted to send but couldn't fit on a vehicle stays in stock,
        #    unshipped, to be reconsidered next round.
        for depot in rotate_depots_for_fairness(world, t):
            transport_capacity = sum(
                tr.capacity for tr in world.transports.values()
                if tr.busy_until <= t and not tr.destroyed
            )
            route_status = {
                shelter.id: not world.route_damaged(depot.position, shelter.position)
                for shelter in world.shelters.values()
            }
            # Phase R: risk score per reachable shelter, so the negotiator
            # and dispatcher can both weigh whether a shelter's need
            # justifies gambling a vehicle on a dangerous-but-passable route.
            route_risk = compute_route_risk_for_depot(world, depot)
            # Phase D: exact operational facts about THIS depot's own supply
            # schedule (not noisy, a real coordinator would know these) plus
            # the trend in reported urgency over recent rounds, so the
            # negotiator can reason about whether to spend freely now or
            # hold some reserve for an anticipated worse round ahead.
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

        total_unmet += unmet_urgent_need(world, arrived_this_step)

        urgency_history.append(round(average_urgency(shelter_urgency_estimates), 3))
        if len(urgency_history) > URGENCY_HISTORY_LENGTH:
            urgency_history.pop(0)

    return total_unmet, tracker


def run_optimizer_baseline(world, rng, n_timesteps: int):
    # Intentionally NOT given Phase A transit delay: the optimizer baseline
    # has always represented an idealized "perfect information, no logistics
    # friction" ceiling (see optimizer/baseline.py's module docstring -- it
    # doesn't model vehicle routing/sequencing at all, that asymmetry with
    # the Dispatcher Agent was already deliberate). Retrofitting a vehicle-
    # routing concept onto the single-shot LP would be a much larger change
    # for a baseline whose whole point is being the theoretical ceiling, not
    # a realistic competitor. This means the agent system is now tested
    # against a bar that's gotten strictly harder (real scarcity AND real
    # transit delay) while the ceiling hasn't moved -- expect gaps to widen
    # somewhat versus pre-Phase-A numbers; that's an honest reflection of
    # how much of the problem "smart routing" actually is, not a bug.
    total_unmet = 0.0
    for t in range(n_timesteps):
        advance_disaster(world, rng)
        alloc = solve_optimal_allocation(world)
        for (depot_id, shelter_id, resource), qty in alloc.items():
            world.depots[depot_id].stock[resource] = max(
                0.0, world.depots[depot_id].stock.get(resource, 0.0) - qty)
        total_unmet += unmet_urgent_need(world, alloc)
    return total_unmet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="haiku_everywhere", choices=list(MODEL_CONFIGS))
    parser.add_argument("--timesteps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")

    # Two independent copies of the *same* starting scenario so the
    # comparison is apples-to-apples.
    world_for_agents = make_scenario(seed=args.seed)
    world_for_optimizer = copy.deepcopy(world_for_agents)
    rng_agents = random.Random(args.seed)
    rng_optimizer = random.Random(args.seed)

    print(f"Running multi-agent system [{args.config}] for {args.timesteps} timesteps...")
    agent_unmet, tracker = run_multi_agent_system(
        world_for_agents, rng_agents, args.timesteps, args.config, api_key)

    print(f"Running centralized optimizer baseline for {args.timesteps} timesteps...")
    optimal_unmet = run_optimizer_baseline(world_for_optimizer, rng_optimizer, args.timesteps)

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Multi-agent system total unmet urgent need: {agent_unmet:.2f}")
    print(f"Optimizer baseline total unmet urgent need: {optimal_unmet:.2f}")
    if optimal_unmet > 0:
        gap_pct = (agent_unmet - optimal_unmet) / optimal_unmet * 100
        print(f"Optimality gap: {gap_pct:+.1f}%  (multi-agent vs. full-information optimal)")
    print()
    print(tracker.summary())


if __name__ == "__main__":
    main()