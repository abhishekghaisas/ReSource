"""
Field Report Agent: turns a shelter's noisy, human-sounding situation report
into a structured urgency estimate (plus a rough population read) the Depot
Agent can negotiate over.

This is deliberately the "simple" extraction task routed to the cheap/local
model tier -- see the cost ablation in run_simulation.py. It never sees the
ground-truth numeric urgency the optimizer baseline uses; it only sees the
same noisy text a real human dispatcher would receive. Population estimate
is extracted the same way and for the same reason: situation_report() text
already includes a noisy population figure (see simulation/world.py), and a
real dispatcher wouldn't have a clean shelter registry to fall back on
mid-crisis either -- so population flows through the same fog-of-war
channel as urgency, rather than being handed to DepotAgent as ground truth.
"""

from __future__ import annotations

import json
import re
from typing import Dict

from agents.base_agent import LLMBackend

SYSTEM_PROMPT = """You are a disaster-relief field report parser. You will receive a short,
informal situation report from a shelter coordinator. Extract:
  - an urgency score from 0.0 (fine) to 1.0 (critical) for each of: food, water, medical.
  - a rough population estimate (the number of people at the shelter), if the report
    mentions one. Use your best guess from context if it's implied but not exact.

Respond with ONLY a JSON object like this, no other text:
{"food": 0.0, "water": 0.0, "medical": 0.0, "population_estimate": 90}
"""

# Used when population can't be extracted from the report at all (parse
# failure, or the model genuinely wasn't given one) -- the midpoint of
# world.py's baseline shelter-population range (30-150), same spirit as the
# 0.5 "neutral guess" fallback already used for urgency. Not recalibrated
# for scenarios that apply a population surge multiplier (see
# run_real_disaster.py); this is a rare fallback path, not the common case.
DEFAULT_POPULATION_ESTIMATE = 90.0


class FieldReportAgent:
    def __init__(self, backend: LLMBackend):
        self.backend = backend

    def parse(self, situation_report: str, timestep: int) -> Dict[str, float]:
        raw = self.backend.call(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=situation_report,
            timestep=timestep,
            max_tokens=120,
        )
        return self._extract_json(raw)

    @staticmethod
    def _extract_json(raw: str) -> Dict[str, float]:
        """
        Small open-weight/local models are less reliable about respecting
        "respond with ONLY json" than Claude is -- this is one of the real
        costs of the cheap tier, flagged explicitly per the plan rather than
        silently patched over. We do a best-effort regex extraction and fall
        back to a neutral guess rather than crashing the simulation.
        """
        match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if not match:
            return {"food": 0.5, "water": 0.5, "medical": 0.5,
                     "population_estimate": DEFAULT_POPULATION_ESTIMATE}
        try:
            parsed = json.loads(match.group(0))
            return {
                "food": float(parsed.get("food", 0.5)),
                "water": float(parsed.get("water", 0.5)),
                "medical": float(parsed.get("medical", 0.5)),
                "population_estimate": max(
                    0.0, float(parsed.get("population_estimate", DEFAULT_POPULATION_ESTIMATE))),
            }
        except (json.JSONDecodeError, TypeError, ValueError):
            return {"food": 0.5, "water": 0.5, "medical": 0.5,
                     "population_estimate": DEFAULT_POPULATION_ESTIMATE}
