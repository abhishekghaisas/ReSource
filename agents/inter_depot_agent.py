"""
Inter-Depot Coordinator Agent (Phase C): decides whether any depot should
transfer surplus stock to another depot this round, using one of the
shared transport vehicles.

Deliberately a SEPARATE, single-purpose agent rather than having DepotAgent
itself bilaterally negotiate with other depots -- letting depots
unilaterally decide to send stock to each other in the same round risks
both simultaneously transferring, or neither noticing the other needs
help, without something seeing the whole picture. One coordinator call per
timestep (not per depot) sees every depot's stock and reachable-shelter
urgency at once and can make a single, globally-informed call.

Simplification: this agent reuses the SAME model tier as depot negotiation
(see the model_name passed in from the caller) rather than being its own
tunable role in MODEL_CONFIGS -- both are similar-shape judgment calls
(weighing competing claims on scarce resources), and adding a 4th
independent role to every config would be a bigger schema change than
this first pass warrants. A dedicated ablation for this specific role is
a reasonable future extension if it turns out to matter.

CONFIRMED (see run_interdepot_test.py and FINDINGS.md): the local model is
NOT suitable for this role. Tested directly with a clear surplus/deficit
scenario (should transfer) and a balanced scenario (should not) -- once a
prompt-echo vehicle-ID bug was fixed, local produced the IDENTICAL transfer
recommendation for both scenarios, meaning it wasn't reasoning about the
input content at all, not just reaching a worse conclusion. Haiku scored
100% correct on both cases across every repeat. This is the third
independent role (after negotiation and dispatch) where local proved
unsuitable for reasoning/comparison tasks while remaining fine for
extraction-only work -- see MODEL_CONFIGS in run_simulation.py for the
resulting architectural principle. Practical consequence: as long as
depot_model is Haiku (true for every MODEL_CONFIGS entry except
local_everywhere, which is a known-non-functional baseline), coordination
correctly gets Haiku too via the reuse above -- no separate action needed
per config, but this reuse is exactly why local_everywhere should not be
treated as a real deployment option for any scenario using 2+ depots.

A transfer physically behaves like a Phase A shipment: real travel time
based on depot-to-depot distance, and it consumes a vehicle from the same
shared fleet shelter deliveries use -- a vehicle moving supplies between
depots is a vehicle NOT delivering to a shelter that round, a genuine and
intentional tradeoff, not a free side-channel.
"""

from __future__ import annotations

import json
import re
from typing import Dict, Optional

from agents.base_agent import LLMBackend
from simulation.world import RESOURCE_TYPES

SYSTEM_PROMPT = """You are an inter-depot coordinator for a disaster relief operation
with multiple depots. Each depot has its own stock and serves a different (possibly
overlapping) set of shelters. Your job is to decide whether ANY depot should transfer
some of its surplus stock to another depot this round, using one of the available
shared transport vehicles.

You are given, per depot: current stock per resource, and a summary of urgency among
the shelters it can CURRENTLY REACH (average urgency 0-1, and how many of those
reachable shelters are critical, i.e. urgency >= 0.5). A depot with LOW average
reachable urgency but HIGH stock of some resource is a transfer CANDIDATE (real
surplus it isn't using). A depot with HIGH average reachable urgency or several
critical reachable shelters but LOW stock of that resource is a transfer TARGET
(real deficit).

A transfer is NOT free: it uses one of the available vehicles for a full round trip,
which means that vehicle can't deliver to a shelter this round either. Only recommend
a transfer if it's clearly worth that tradeoff -- e.g. one depot is sitting on medical
supplies it doesn't need while shelters only reachable from another, medical-short
depot are critical. Do not recommend a transfer that would leave the source depot
unable to cover its own reachable shelters' needs. If BOTH depots show similar stock
and similar reachable urgency, there is no clear surplus or deficit -- say so; most
rounds, and any round where the depots look roughly balanced, should have NO transfer.

Each depot's summary may also include "transfer_risk" -- a 0.0-1.0 score per OTHER
depot, reflecting how dangerous the route between them currently is (e.g. passing near
a spreading fire). This is NOT the same as being unreachable -- a risky transfer route
is still possible, but a real gamble: the shipment may arrive fine, arrive partially,
or be lost completely along with the vehicle. Only route a transfer through real risk
if the deficit being addressed is severe enough to justify it (e.g. a depot completely
cut off from all its own shelters, as in a genuine emergency consolidation) -- don't
risk a vehicle moving stock between depots over a marginal imbalance.

CRITICAL formatting rule: "vehicle_id" MUST be copied EXACTLY from the
"available_vehicles" list you are given (e.g. if available_vehicles contains "T1" and
"T2", your vehicle_id must be "T1" or "T2") -- never invent a vehicle ID that isn't
in that list, even if it looks like a plausible name.

Respond with ONLY a JSON object, one of these two shapes:
{"transfer": {"source_depot_id": "D1", "dest_depot_id": "D2", "resource": "medical",
"quantity": 20.0, "vehicle_id": "T1"}}
or, if no transfer should happen this round:
{"transfer": null}
No other text.
"""


class InterDepotCoordinatorAgent:
    def __init__(self, backend: LLMBackend):
        self.backend = backend

    def coordinate(self, depot_summaries: Dict[str, dict],
                    available_vehicles: Dict[str, dict], timestep: int) -> Optional[dict]:
        """
        depot_summaries: {depot_id: {"stock": {resource: qty},
                            "avg_reachable_urgency": float,
                            "n_critical_reachable_shelters": int,
                            "transfer_risk": {other_depot_id: float 0-1}}}
        (urgency figures here are aggregated from FieldReportAgent's noisy
        extraction, same fog-of-war status as everywhere else it's used --
        not ground truth. transfer_risk, like route_status/route_risk
        elsewhere, is exact -- Phase R. The "transfer_risk" key is optional
        per depot; its absence means no meaningful risk to any other depot
        this round.)

        available_vehicles: {vehicle_id: {"capacity": float, "speed": int}}
        -- vehicles not currently busy AND not destroyed (Phase R: a
        vehicle lost on a prior risky route never reappears here).

        Returns: {"source_depot_id", "dest_depot_id", "resource",
                  "quantity", "vehicle_id"} or None if no transfer.
        """
        if len(depot_summaries) < 2 or not available_vehicles:
            # Nothing to coordinate with only one depot, or no vehicle free
            # to carry a transfer even if one were worthwhile -- skip the
            # LLM call entirely rather than asking a question with no
            # possible useful answer.
            return None

        prompt = json.dumps({
            "depot_summaries": depot_summaries,
            "available_vehicles": available_vehicles,
        }, indent=2)

        raw = self.backend.call(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
            timestep=timestep,
            max_tokens=400,
        )
        return self._extract(raw, depot_summaries, available_vehicles)

    @staticmethod
    def _extract(raw: str, depot_summaries: Dict[str, dict],
                 available_vehicles: Dict[str, dict]) -> Optional[dict]:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

        transfer = parsed.get("transfer")
        if not transfer or not isinstance(transfer, dict):
            return None

        # Defense-in-depth validation, same philosophy as DepotAgent's and
        # DispatcherAgent's clamps: don't trust prompt compliance alone.
        source = transfer.get("source_depot_id")
        dest = transfer.get("dest_depot_id")
        resource = transfer.get("resource")
        vehicle_id = transfer.get("vehicle_id")
        qty = transfer.get("quantity")

        if source not in depot_summaries or dest not in depot_summaries or source == dest:
            return None
        if vehicle_id not in available_vehicles:
            return None
        if resource not in RESOURCE_TYPES:
            return None
        try:
            qty = max(0.0, float(qty))
        except (TypeError, ValueError):
            return None
        if qty <= 0:
            return None

        # Clamp to what the source actually has and what the vehicle can
        # carry -- can't transfer more than either allows, regardless of
        # what the model asked for.
        available_stock = depot_summaries[source]["stock"].get(resource, 0.0)
        vehicle_capacity = available_vehicles[vehicle_id]["capacity"]
        qty = min(qty, available_stock, vehicle_capacity)
        if qty <= 0:
            return None

        return {
            "source_depot_id": source,
            "dest_depot_id": dest,
            "resource": resource,
            "quantity": qty,
            "vehicle_id": vehicle_id,
        }
