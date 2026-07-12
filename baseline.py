"""
Centralized, full-information optimal allocation baseline.

This is the "God's-eye-view" comparison point: given perfect knowledge of
every shelter's true urgency, every depot's stock, and every route's
condition, what is the mathematically optimal allocation of resources this
timestep? The multi-agent system never gets to see this -- it only has
noisy situation reports and partial negotiation -- so the gap between this
and the multi-agent outcome is your core evaluation metric.

Uses scipy.optimize.linprog (HiGHS solver, ships with scipy -- no external
solver binary or extra install needed, unlike PuLP/CBC).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linprog

from simulation.world import WorldState, RESOURCE_TYPES, RESOURCE_CRITICALITY, route_risk_score


def solve_optimal_allocation(world: WorldState) -> Dict[Tuple[str, str, str], float]:
    """
    Solve: maximize total urgency-weighted resource delivered this timestep,
    subject to depot stock limits, per-shelter reasonable-use caps, and total
    transport capacity.

    Route damage and travel time are simplified into a feasibility flag per
    (depot, shelter) pair for this single-timestep LP; the Dispatcher Agent
    in the multi-agent system handles the harder multi-timestep
    routing/sequencing problem, which is intentionally *not* modeled here --
    that asymmetry is part of what the comparison is measuring.

    Returns: dict mapping (depot_id, shelter_id, resource) -> quantity allocated.
    """
    depots = list(world.depots.values())
    shelters = list(world.shelters.values())

    var_keys: List[Tuple[str, str, str]] = []
    for depot in depots:
        for shelter in shelters:
            for resource in RESOURCE_TYPES:
                var_keys.append((depot.id, shelter.id, resource))
    n = len(var_keys)
    index_of = {k: i for i, k in enumerate(var_keys)}

    # linprog minimizes, so negate the (urgency, population, and criticality
    # weighted) objective to maximize it.
    #
    # FIX: this previously used `1.0 + 5.0*urgency` alone, with NO population
    # or resource-criticality weighting -- while unmet_urgent_need() (the
    # actual evaluation metric, below) has always been population-weighted
    # and is now also criticality-weighted. That mismatch meant the "optimal"
    # baseline wasn't actually optimizing for what gets measured: a unit
    # delivered to a 500-person shelter scored the same in this objective as
    # one delivered to a 50-person shelter, even though the eval metric would
    # credit the former ~10x more. Fixed so the marginal reward per unit
    # delivered here matches the marginal reduction in unmet_urgent_need for
    # that same unit (shortfall decreases 1-for-1 with delivered, up to the
    # need cap), keeping the "optimal" baseline genuinely optimal with
    # respect to what's actually scored. The `1.0 +` offset is preserved
    # from the original design intent: give some baseline value to
    # low-urgency delivery too, rather than zero incentive to help any
    # shelter that isn't already critical.
    c = np.zeros(n)
    for (depot_id, shelter_id, resource), i in index_of.items():
        shelter = world.shelters[shelter_id]
        criticality = RESOURCE_CRITICALITY.get(resource, 1.0)
        base_value = (1.0 + 5.0 * shelter.urgency(resource)) * criticality * (shelter.population / 100.0)
        # Phase R: a route in the risk buffer (not hard-blocked, but not
        # fully safe either) isn't a sure thing anymore -- the multi-agent
        # simulation now actually rolls an outcome (see
        # resolve_risk_outcome()) that can partially or totally lose a
        # risky shipment. The LP is a single deterministic solve, not a
        # simulation, so it can't gamble the same way; instead its
        # objective value for a risky route is discounted by expected
        # surviving fraction, keeping it a fair (still idealized, still not
        # literally simulating outcomes) ceiling rather than one that gets
        # to treat every non-hard-blocked route as a sure thing while the
        # agents face real loss risk on the same route.
        depot = world.depots[depot_id]
        risk = route_risk_score(world, depot.position, shelter.position)
        expected_surviving_fraction = 1.0 - risk * 0.5  # matches resolve_risk_outcome's expected loss
        c[i] = -base_value * expected_surviving_fraction

    # Bounds: 0 <= x <= 0 for infeasible (route-damaged) pairs, else 0 <= x <= inf.
    bounds = []
    for (depot_id, shelter_id, resource), i in index_of.items():
        depot = world.depots[depot_id]
        shelter = world.shelters[shelter_id]
        feasible = not world.route_damaged(depot.position, shelter.position)
        bounds.append((0, 0) if not feasible else (0, None))

    A_ub = []
    b_ub = []

    # Depot stock constraints.
    for depot in depots:
        for resource in RESOURCE_TYPES:
            row = np.zeros(n)
            for shelter in shelters:
                row[index_of[(depot.id, shelter.id, resource)]] = 1.0
            A_ub.append(row)
            b_ub.append(depot.stock.get(resource, 0.0))

    # Reasonable-use cap per shelter/resource (avoid dumping everything at one shelter).
    for shelter in shelters:
        for resource in RESOURCE_TYPES:
            row = np.zeros(n)
            for depot in depots:
                row[index_of[(depot.id, shelter.id, resource)]] = 1.0
            A_ub.append(row)
            reasonable_cap = shelter.consumption_rate.get(resource, 1.0) * 5 + 10
            b_ub.append(reasonable_cap)

    # Total transport capacity constraint across all goods moved this timestep.
    total_transport_capacity = sum(t.capacity for t in world.transports.values())
    A_ub.append(np.ones(n))
    b_ub.append(total_transport_capacity)

    result = linprog(
        c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        # Infeasible edge case (e.g. all routes blocked): no allocation rather
        # than crashing -- this is itself a valid, informative outcome.
        return {k: 0.0 for k in var_keys}

    return {k: max(0.0, float(result.x[i])) for k, i in index_of.items()}


def unmet_urgent_need(world: WorldState, allocation: Dict[Tuple[str, str, str], float],
                       urgency_threshold: float = 0.5) -> float:
    """
    Evaluation metric: total unmet need (population- and criticality-
    weighted) at shelters whose urgency exceeds the threshold, after
    applying the given allocation. Lower is better. Used identically for
    both the optimizer's own output and the multi-agent system's output, so
    they're directly comparable.

    NOTE: adding RESOURCE_CRITICALITY weighting here (on top of the
    population weighting that was already present) changes the absolute
    scale of this metric versus prior runs -- unmet medical need is now
    weighted 3x, water 2x, food 1x, rather than all three counting equally.
    Absolute unmet-need numbers from before this change are not directly
    comparable to numbers after it; percentage optimality gaps are still
    comparable in spirit (both sides of the gap use the same metric
    definition within a single run) but any historical run should be
    re-generated rather than compared number-for-number against this
    version.
    """
    delivered_per_shelter_resource: Dict[Tuple[str, str], float] = {}
    for (depot_id, shelter_id, resource), qty in allocation.items():
        key = (shelter_id, resource)
        delivered_per_shelter_resource[key] = delivered_per_shelter_resource.get(key, 0.0) + qty

    unmet = 0.0
    for shelter in world.shelters.values():
        for resource in RESOURCE_TYPES:
            urgency = shelter.urgency(resource)
            if urgency >= urgency_threshold:
                delivered = delivered_per_shelter_resource.get((shelter.id, resource), 0.0)
                need = shelter.consumption_rate.get(resource, 0.0) * 5  # 5-timestep buffer target
                shortfall = max(0.0, need - delivered)
                criticality = RESOURCE_CRITICALITY.get(resource, 1.0)
                unmet += shortfall * urgency * criticality * (shelter.population / 100.0)
    return unmet