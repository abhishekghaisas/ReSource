"""
Depot Agent: receives (noisy-urgency-derived) requests from every shelter
and decides how to allocate scarce stock among them. This is the one place
in the system where real judgment matters -- weighing competing claims,
resisting an inflated request, being fair across rounds -- so it's the
agent this project's model-tier ablation (Haiku vs Sonnet vs local-only)
is centered on.
"""

from __future__ import annotations

import json
import re
from typing import Dict, Optional

from agents.base_agent import LLMBackend
from simulation.world import RESOURCE_CRITICALITY, RESOURCE_TYPES

SYSTEM_PROMPT = f"""You are a disaster-relief depot coordinator. Multiple shelters are
requesting resources you have limited stock of. You must decide how much of each
resource to send to each shelter this round, given:
  - Each shelter's reported urgency (0-1) per resource, self-reported and possibly
    inflated -- use judgment, don't just believe the highest number blindly.
  - Each shelter's reported population estimate -- also self-reported/approximate,
    not exact. All else equal, prioritize the shelter where more people are affected:
    delivering the same resource to a shelter of 500 people helps more people than
    delivering it to a shelter of 50, even at similar urgency.
  - Fixed relative criticality across resource types (not shelter-reported, this is
    standard triage priority): medical={RESOURCE_CRITICALITY['medical']}, water={RESOURCE_CRITICALITY['water']},
    food={RESOURCE_CRITICALITY['food']}. A medical shortage is more urgent than an
    equally-scored food shortage -- weigh accordingly when stock is tight enough that
    you can't fully cover every request.
  - Each shelter's route status FROM THIS DEPOT this round: "reachable" or "blocked".
    Do not allocate anything to a shelter marked "blocked" -- the route is currently
    impassable and nothing you send can arrive, so that stock would simply be wasted
    instead of helping a shelter you CAN actually reach. Route status can differ by
    depot and by round as damage spreads or clears; always check the current value,
    don't assume from a prior round.
  - (When provided) A "route_risk" score (0.0-1.0) for each REACHABLE shelter. This is
    NOT the same as "blocked" -- a risky route is still passable, but attempting it is
    a real gamble: higher risk means a real chance the shipment is partially lost, or
    the vehicle and its full cargo are lost entirely, not delivered at all. A risk near
    0 is essentially safe. A risk near 1 means the route runs close to real danger
    (e.g. near a spreading fire) -- only worth committing scarce stock and a vehicle to
    a shelter that risky if its urgency (and population) genuinely justify the gamble.
    Don't send resources into high risk for a shelter that isn't actually critical --
    that's wasting stock and a vehicle you could use safely elsewhere. Not present or
    0 for a shelter means no meaningful risk on that route this round.
  - Your current stock per resource.
  - Total transport capacity available this round (you cannot exceed it in total
    units shipped across all shelters/resources).
  - (When provided) A lookahead_context block with two things you can use to decide
    whether to spend generously now or hold some reserve: "timesteps_until_next_resupply"
    (an exact count -- 0 means your next resupply convoy arrives after this round's
    decision) and "expected_resupply_fraction" (the fraction of your ORIGINAL stock
    it will add), plus "recent_avg_urgency_trend" (the last few rounds' average
    shelter urgency, oldest first) so you can tell whether the situation is
    escalating or stabilizing. If resupply is imminent and urgency is stable, it's
    reasonable to spend more freely. If resupply is many rounds away and urgency is
    trending up, holding back some reserve for a worse round ahead can be the better
    call -- but don't starve a currently-critical shelter just to hoard for a
    hypothetical future one; this is a secondary consideration, not a reason to
    ignore present urgency.

Be fair but prioritize genuine urgency, scaled by population and resource criticality,
among shelters you can actually reach. Do not starve a reachable shelter completely if
you can help it, unless stock is truly exhausted.

Respond with ONLY a JSON object mapping shelter_id -> {{resource: quantity}} (food/water/
medical only -- population_estimate, route_status, route_risk, and lookahead_context are
context, not something to allocate; omit or zero out any shelter marked "blocked"), e.g.:
{{"S1": {{"food": 5.0, "water": 3.0, "medical": 0.0}}, "S2": {{...}}}}
No other text.
"""


class DepotAgent:
    def __init__(self, backend: LLMBackend):
        self.backend = backend

    def allocate(self, depot_id: str, stock: Dict[str, float],
                 transport_capacity: float,
                 shelter_requests: Dict[str, Dict[str, float]],
                 timestep: int,
                 route_status: Optional[Dict[str, bool]] = None,
                 lookahead_context: Optional[Dict] = None,
                 route_risk: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, float]]:
        """
        shelter_requests: {shelter_id: {resource: urgency_0_to_1, ...,
                            "population_estimate": float}}
        (population_estimate comes from FieldReportAgent's noisy extraction,
        same fog-of-war status as urgency -- not ground truth.)

        route_status: {shelter_id: True/False} -- True if the route from
        THIS depot to that shelter is currently passable, False if blocked.
        Unlike urgency/population, this is exact (mirrors what the optimizer
        baseline already uses via world.route_damaged()), not noisy --
        route condition is the kind of thing a real dispatcher would know
        directly (road closure reports), not something inferred from a
        shelter's self-report. Defaults to None, meaning "treat every
        shelter as reachable" -- preserves old behavior for any caller that
        hasn't been updated to pass this yet.

        route_risk: optional {shelter_id: float 0-1} -- Phase R. Risk score
        for a REACHABLE (not hard-blocked) route, reflecting proximity to a
        spreading hazard (e.g. wildfire). Exact, like route_status, not
        noisy. Defaults to None, meaning no route in this call carries
        elevated risk (preserves old behavior for any caller that hasn't
        been updated to pass this).

        lookahead_context: optional dict with
          {"timesteps_until_next_resupply": int,
           "expected_resupply_fraction": float,
           "recent_avg_urgency_trend": List[float]}
        -- exact operational facts (a real coordinator would know their own
        supply schedule), not noisy like urgency/population. Defaults to
        None, meaning no lookahead info is given (preserves old myopic
        behavior for any caller that hasn't been updated to pass this).

        Returns: {shelter_id: {resource: quantity_allocated}}
        """
        route_status = route_status or {sid: True for sid in shelter_requests}
        payload = {
            "depot_id": depot_id,
            "stock": stock,
            "transport_capacity": transport_capacity,
            "shelter_urgency_reports": shelter_requests,
            "route_status": {sid: ("reachable" if ok else "blocked")
                              for sid, ok in route_status.items()},
        }
        if route_risk is not None:
            payload["route_risk"] = {sid: round(r, 3) for sid, r in route_risk.items() if r > 0}
        if lookahead_context is not None:
            payload["lookahead_context"] = lookahead_context
        prompt = json.dumps(payload, indent=2)

        raw = self.backend.call(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
            timestep=timestep,
            # Higher than the other agents' budgets: negotiation is the one
            # task where the model may use extended/adaptive thinking before
            # answering, and a too-small max_tokens can get entirely consumed
            # by that reasoning, leaving no room for the actual JSON output
            # (confirmed via response.stop_reason -- see base_agent.py).
            max_tokens=1500,
        )
        return self._extract_and_validate(raw, stock, transport_capacity, shelter_requests,
                                            route_status)

    @staticmethod
    def _extract_and_validate(raw: str, stock: Dict[str, float], transport_capacity: float,
                                shelter_requests: Dict[str, Dict[str, float]],
                                route_status: Optional[Dict[str, bool]] = None
                                ) -> Dict[str, Dict[str, float]]:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        allocation: Dict[str, Dict[str, float]] = {}
        if match:
            try:
                allocation = json.loads(match.group(0))
            except json.JSONDecodeError:
                allocation = {}

        # FIX (see project diagnostics): the original clamp walked
        # `shelter_requests` in fixed dict-insertion order (always S1, S2,
        # ..., S10) and subtracted each shelter's request from remaining
        # stock/capacity in that order. Whenever total requests exceeded
        # what was available -- which happens routinely under real scarcity,
        # not just as an edge case -- the shelter processed FIRST got 100%
        # of its request and every shelter after it got 0%, regardless of
        # actual urgency or what the model intended. This was a silent,
        # purely positional override of the negotiator's decision: confirmed
        # via a direct unit test where 10 equally-urgent shelters with
        # identical requests produced "S1 gets everything, S2-S10 get
        # nothing" independent of which model (or no model at all) produced
        # the allocation. It's also the leading suspect for why local/Haiku/
        # mixed configs converged to near-identical outcomes on the
        # real-disaster scenario: once stock is scarce enough that shelters
        # compete most timesteps, this clamp -- not the negotiator -- was
        # deciding who gets fed.
        #
        # Fix: clamp PROPORTIONALLY instead of sequentially. If a resource's
        # total requested amount exceeds available stock, scale every
        # shelter's request down by the same factor rather than fully
        # satisfying some and zeroing others by dict position. This
        # preserves whatever relative prioritization the negotiator actually
        # chose (a shelter requesting 2x more than another still gets ~2x
        # more after scaling) instead of overriding it with iteration order.
        route_status = route_status or {sid: True for sid in shelter_requests}
        raw_requested: Dict[str, Dict[str, float]] = {}
        for shelter_id in shelter_requests:
            requested = allocation.get(shelter_id, {})
            # Defense-in-depth (same reasoning as the clamp fix above): don't
            # rely on the model to always honor "skip blocked shelters" from
            # the prompt alone. A blocked shelter's request is zeroed out
            # HERE, before it can compete for stock at all -- this both
            # guarantees no stock is ever spent on an undeliverable route
            # regardless of model compliance, and frees that stock up for
            # reachable shelters instead of it sitting reserved for someone
            # who can't receive it this round.
            reachable = route_status.get(shelter_id, True)
            if not reachable:
                raw_requested[shelter_id] = {r: 0.0 for r in RESOURCE_TYPES}
            else:
                raw_requested[shelter_id] = {
                    r: max(0.0, float(requested.get(r, 0.0))) for r in RESOURCE_TYPES
                }

        clamped: Dict[str, Dict[str, float]] = {sid: {} for sid in shelter_requests}
        for resource in RESOURCE_TYPES:
            total_requested = sum(raw_requested[sid][resource] for sid in shelter_requests)
            available = max(0.0, stock.get(resource, 0.0))
            scale = 1.0 if total_requested <= available or total_requested == 0 else available / total_requested
            for sid in shelter_requests:
                clamped[sid][resource] = raw_requested[sid][resource] * scale

        # Total transport capacity is a single pooled constraint across all
        # resources/shelters (not per-resource), so it's applied as a second
        # proportional pass over whatever the per-resource pass produced --
        # same reasoning: scale everyone down together rather than let
        # iteration order decide who gets cut first.
        total_allocated = sum(clamped[sid][r] for sid in shelter_requests for r in RESOURCE_TYPES)
        if total_allocated > transport_capacity and total_allocated > 0:
            cap_scale = transport_capacity / total_allocated
            for sid in shelter_requests:
                for r in RESOURCE_TYPES:
                    clamped[sid][r] *= cap_scale

        return clamped