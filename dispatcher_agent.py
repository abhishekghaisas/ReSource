"""
Dispatcher Agent: given ONE depot's allocation decision this round, decides
which available vehicle carries which delivery. This is a mechanical,
mostly-arithmetic task -- another candidate for the cheap/local model tier
rather than a frontier model.

Phase A change: this agent's output now actually controls what happens in
the simulation. Previously `sequence()` was called once per timestep with
all depots' allocations flattened together (losing which depot each item
came from whenever two depots targeted the same shelter+resource), and its
return value was discarded entirely -- deliveries were treated as arriving
instantly regardless of what the Dispatcher decided. Now:
  - It's called once PER DEPOT (natural, since a vehicle's travel time
    depends on which depot it starts from -- can't compute that from a
    flattened multi-depot dict).
  - Its vehicle assignments determine which deliveries are actually shipped
    this round (see run_simulation.py) -- only the shipped quantity leaves
    depot.stock; whatever doesn't get assigned to a vehicle stays in stock,
    unshipped, and is reconsidered next round.
  - Each shipment gets a real arrival time from simulation.world.dispatch_shipment()
    (grid distance / vehicle speed), not instant delivery.

Simplification: each vehicle carries AT MOST one delivery (one shelter, one
round trip) per dispatch call. Real multi-stop routing (one truck visiting
several shelters in sequence) would need real route-distance calculation,
not just point-to-point manhattan distance -- out of scope for this phase.
If the model assigns a vehicle to multiple deliveries in one round, only
the first is kept; the rest are dropped (see _extract).
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from agents.base_agent import LLMBackend
from simulation.world import WorldState

SYSTEM_PROMPT = """You are a logistics dispatcher for ONE depot. You have that depot's
allocation decision (how much of each resource should go to each shelter this round)
and a fleet of available transport vehicles. Some routes are damaged/blocked. Vehicles
travel at different speeds, so farther shelters take longer to reach -- that's fine,
just don't assign a delivery to a shelter whose route is currently blocked.

Assign each delivery to ONE vehicle, respecting:
  - Vehicle capacity (total units carried per trip) -- do not overload a vehicle.
  - Each vehicle can only be assigned ONE delivery this round (it needs to complete
    the round trip before it can be reloaded) -- don't give one vehicle multiple stops.
  - Do not route through blocked routes.

If total allocated goods exceed what the available fleet can carry this round,
prioritize higher-urgency, higher-population deliveries and leave the rest
unshipped for next round (it will simply remain in depot stock).

(When provided) A "route_risk" score (0.0-1.0) per shelter -- NOT the same as blocked.
A risky route is still passable, but sending a vehicle down one is a real gamble: it
may arrive fine, arrive with only part of its cargo, or be lost completely along with
the vehicle itself (permanently removing that vehicle from the fleet). Weigh this
against the shelter's urgency/population in shelter_urgency -- committing a vehicle to
a high-risk route only makes sense if that shelter's need genuinely justifies the
gamble. For a low-urgency shelter, prefer sending it via a safer route or holding the
delivery rather than risking a vehicle you may need later. Not present or 0 means no
meaningful risk on that route this round.

CRITICAL formatting rules:
  - Every key in your response MUST be an ACTUAL vehicle ID copied from the
    "available_vehicles" list you're given (e.g. "T1", "T2", "T3") -- never use the
    literal word "vehicle_id" as a key, that is only a name for the field, not a value.
  - Only include vehicles you are actually dispatching this round. Do not list a
    vehicle with an empty delivery, and do not repeat the same key twice.
  - Keep your answer as short as possible: one entry per vehicle, no extra text,
    no explanation.

Example, if available_vehicles contains "T1" and "T2", and you decide to send T1 to
S3 with 4.0 food and leave T2 unused this round, respond with EXACTLY this shape
(using your own real shelter/resource/quantity values):
{"T1": [{"shelter_id": "S3", "resource": "food", "quantity": 4.0}]}

Respond with ONLY the JSON object. No other text, no markdown formatting.
"""


class DispatcherAgent:
    def __init__(self, backend: LLMBackend):
        self.backend = backend

    def sequence(self, world: WorldState, depot_id: str,
                 depot_allocation: Dict[str, Dict[str, float]],
                 shelter_urgency: Dict[str, Dict[str, float]], timestep: int,
                 route_risk: Optional[Dict[str, float]] = None
                 ) -> Dict[str, List[dict]]:
        """
        depot_allocation: {shelter_id: {resource: quantity}} -- THIS depot's
        allocation only (from a single DepotAgent.allocate() call), not
        flattened across multiple depots.

        route_risk: optional {shelter_id: float 0-1} -- Phase R. Risk score
        for a reachable (not hard-blocked) route from this depot to that
        shelter. Exact, not noisy. Defaults to None, meaning no route in
        this call carries elevated risk (preserves old behavior for any
        caller that hasn't been updated to pass this).

        Returns: {vehicle_id: [{"shelter_id", "resource", "quantity"}]}
        (each vehicle's list has at most one entry after validation --
        see _extract).
        """
        depot = world.depots[depot_id]
        vehicles = {
            t.id: {"capacity": t.capacity, "speed": t.speed}
            for t in world.transports.values()
            # Phase R: a destroyed vehicle (lost on a prior risky route) is
            # gone permanently -- never available again, not just busy.
            if t.busy_until <= timestep and not t.destroyed
        }
        shelter_positions = {
            sid: list(world.shelters[sid].position) for sid in depot_allocation
            if sid in world.shelters
        }

        payload = {
            "depot_id": depot_id,
            "depot_position": list(depot.position),
            "depot_allocation": depot_allocation,
            "shelter_positions": shelter_positions,
            "shelter_urgency": shelter_urgency,
            "available_vehicles": vehicles,
            "blocked_routes": list(world.blocked_routes),
        }
        if route_risk is not None:
            payload["route_risk"] = {sid: round(r, 3) for sid, r in route_risk.items() if r > 0}
        prompt = json.dumps(payload, indent=2, default=list)

        raw = self.backend.call(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
            timestep=timestep,
            # Raised from 500: the local model was observed truncating
            # mid-response on this task's now-larger structured output
            # (multiple vehicles, nested delivery objects) -- confirmed via
            # --debug output in run_shipment_trace.py showing cut-off JSON
            # like `"resource":` with nothing after it. Unlike the Anthropic
            # backend, the local backend has no retry-with-bigger-budget
            # logic for truncation, so getting the budget right up front
            # matters more here.
            max_tokens=900,
        )
        return self._extract(raw, vehicles, depot_allocation, world, depot)

    @staticmethod
    def _extract(raw: str, vehicles: Dict, depot_allocation: Dict[str, Dict[str, float]],
                 world: WorldState, depot) -> Dict[str, List[dict]]:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}

        # Pass 1: drop hallucinated vehicles, keep only the FIRST delivery
        # per vehicle (single-stop-per-round simplification -- see module
        # docstring), drop deliveries to shelters not in this depot's own
        # allocation (hallucination guard) or whose route is blocked from
        # this depot (defense-in-depth, same reasoning as DepotAgent's
        # route-status masking: don't trust prompt compliance alone).
        candidate: Dict[str, dict] = {}
        for vid, deliveries in parsed.items():
            # Defensive: reject the literal placeholder string even if a
            # weaker model still echoes it from the prompt's example
            # occasionally -- confirmed as a real failure mode with the
            # local model (it used "vehicle_id" as a literal key instead of
            # substituting a real vehicle ID like "T1").
            if vid == "vehicle_id":
                continue
            if vid not in vehicles or not isinstance(deliveries, list) or not deliveries:
                continue
            delivery = deliveries[0]
            sid = delivery.get("shelter_id")
            resource = delivery.get("resource")
            qty = delivery.get("quantity")
            if sid not in depot_allocation or resource is None or qty is None:
                continue
            shelter = world.shelters.get(sid)
            if shelter is None or world.route_damaged(depot.position, shelter.position):
                continue
            qty = max(0.0, float(qty))
            if qty <= 0:
                continue
            candidate[vid] = {"shelter_id": sid, "resource": resource, "quantity": qty}

        if not candidate:
            return {}

        # Pass 2: a (shelter, resource) pair might get claimed by more than
        # one vehicle if the model double-books it -- cap the combined total
        # at what DepotAgent actually allocated for that pair, scaling down
        # proportionally rather than picking a winner by iteration order
        # (same lesson as DepotAgent's own clamp fix).
        requested_by_pair: Dict[tuple, float] = {}
        for vid, d in candidate.items():
            key = (d["shelter_id"], d["resource"])
            requested_by_pair[key] = requested_by_pair.get(key, 0.0) + d["quantity"]

        for vid, d in candidate.items():
            key = (d["shelter_id"], d["resource"])
            allocated_cap = depot_allocation.get(d["shelter_id"], {}).get(d["resource"], 0.0)
            total_requested = requested_by_pair[key]
            if total_requested > allocated_cap and total_requested > 0:
                d["quantity"] *= allocated_cap / total_requested

        # Pass 3: cap each vehicle at its own capacity (single delivery per
        # vehicle this round, so this is a direct clamp, not proportional --
        # there's nothing else on that vehicle to scale against).
        for vid, d in candidate.items():
            cap = vehicles[vid]["capacity"]
            d["quantity"] = min(d["quantity"], cap)

        return {vid: [d] for vid, d in candidate.items() if d["quantity"] > 0}