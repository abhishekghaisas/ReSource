"""
Base agent infrastructure: pluggable model backends + cost tracking.

The whole point of this file is to make the cost-vs-quality experiment
(local model vs Haiku vs Sonnet, per agent role) a one-line config change,
not a rewrite. Every agent call goes through `LLMBackend.call()`, which
logs tokens/cost centrally regardless of which backend served the request.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Pricing table (USD per million tokens). Update as prices change --
# see the project README for how this feeds the cost-feasibility report.
# ---------------------------------------------------------------------------
PRICING = {
    # Anthropic hosted models (input, output) per 1M tokens.
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-opus-4-8": (5.00, 25.00),
    # Local models: zero marginal API cost. Kept as a field (not omitted)
    # so the cost report can show "$0.00 -- local inference" explicitly
    # rather than silently excluding these calls.
    "local": (0.0, 0.0),
}

# Process-wide memory of which models have rejected an explicit temperature
# override (observed: claude-sonnet-5, likely due to forced extended
# thinking). Populated lazily on first failure -- see LLMBackend._call_anthropic.
_TEMPERATURE_UNSUPPORTED_MODELS: set = set()


@dataclass
class CallRecord:
    agent_role: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    timestep: int


@dataclass
class CostTracker:
    records: List[CallRecord] = field(default_factory=list)
    # Not a dataclass field with type annotation -- threading.Lock isn't a
    # type dataclass can compare/repr sensibly, so it's attached in
    # __post_init__ instead. Needed because run_real_disaster.py (and any
    # other caller) may now fire multiple LLM calls concurrently via a
    # thread pool -- list.append is safe under CPython's GIL for the simple
    # case, but the read-modify-write pattern elsewhere (total_cost,
    # cost_by_role) reading `records` while another thread appends is worth
    # guarding explicitly rather than relying on GIL implementation details.
    def __post_init__(self):
        self._lock = threading.Lock()

    def log(self, agent_role: str, model: str, input_tokens: int, output_tokens: int,
             latency_s: float, timestep: int):
        in_price, out_price = PRICING.get(model, (0.0, 0.0))
        cost = (input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price
        record = CallRecord(agent_role, model, input_tokens, output_tokens,
                             cost, latency_s, timestep)
        with self._lock:
            self.records.append(record)

    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self.records)

    def cost_by_role(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for r in self.records:
            out[r.agent_role] = out.get(r.agent_role, 0.0) + r.cost_usd
        return out

    def summary(self) -> str:
        lines = [f"Total cost: ${self.total_cost():.4f} across {len(self.records)} calls"]
        for role, cost in sorted(self.cost_by_role().items(), key=lambda kv: -kv[1]):
            n_calls = sum(1 for r in self.records if r.agent_role == role)
            lines.append(f"  {role:20s}  ${cost:.4f}  ({n_calls} calls)")
        return "\n".join(lines)


class LLMBackend:
    """
    Thin, uniform interface over:
      - Anthropic API (Claude Haiku/Sonnet/Opus) for judgment-heavy agents
      - Local Ollama/MLX server for structured-extraction agents

    Swapping an agent's model tier is a one-line change at construction time,
    which is what makes the Haiku-vs-Sonnet and local-vs-hosted ablations
    (see run_simulation.py --model-config) cheap to run.
    """

    def __init__(self, model: str, tracker: CostTracker, agent_role: str,
                 anthropic_api_key: Optional[str] = None,
                 local_endpoint: str = "http://localhost:11434/api/generate"):
        self.model = model
        self.tracker = tracker
        self.agent_role = agent_role
        self.anthropic_api_key = anthropic_api_key
        self.local_endpoint = local_endpoint

    def call(self, system_prompt: str, user_prompt: str, timestep: int,
              max_tokens: int = 300) -> str:
        start = time.time()

        if self.model == "local":
            text, in_tok, out_tok = self._call_local(system_prompt, user_prompt, max_tokens)
        else:
            text, in_tok, out_tok = self._call_anthropic(system_prompt, user_prompt, max_tokens)

        latency = time.time() - start
        self.tracker.log(self.agent_role, self.model, in_tok, out_tok, latency, timestep)
        return text

    # -- Anthropic backend --------------------------------------------------
    def _call_anthropic(self, system_prompt: str, user_prompt: str, max_tokens: int):
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError(
                "pip install anthropic --break-system-packages to use hosted Claude models"
            ) from e

        client = anthropic.Anthropic(api_key=self.anthropic_api_key)

        # Retry with escalating token budget if extended thinking consumes
        # the entire max_tokens before producing any text. Confirmed real
        # failure mode (see project history): stop_reason == "max_tokens"
        # with only a "thinking" content block and no "text" block. A single
        # fixed max_tokens value isn't reliable because thinking length
        # varies with how complex the negotiation decision is that round.
        budgets_to_try = [max_tokens, max_tokens * 2, max_tokens * 4]
        response = None
        text = ""
        total_input_tokens = 0
        total_output_tokens = 0

        for attempt, budget in enumerate(budgets_to_try):
            request_kwargs = dict(
                model=self.model,
                max_tokens=budget,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            # Some models (observed: claude-sonnet-5, likely due to forced
            # extended thinking) reject an explicit temperature override with
            # a 400 error ("temperature is deprecated for this model") rather
            # than just ignoring it. Rather than hardcode a model list that
            # will go stale, remember per-model (process-wide) whether this
            # failed before, and fall back to omitting temperature entirely --
            # this means we lose determinism for those specific models, which
            # is exactly why --repeats in run_sweep.py exists: to quantify
            # whatever sampling variance we can't eliminate.
            if self.model not in _TEMPERATURE_UNSUPPORTED_MODELS:
                request_kwargs["temperature"] = 0.0

            try:
                response = client.messages.create(**request_kwargs)
            except anthropic.BadRequestError as e:
                if "temperature" in str(e).lower() and "temperature" in request_kwargs:
                    print(f"  [INFO] {self.model} rejected explicit temperature; "
                          f"retrying without it (results for this model will not "
                          f"be fully deterministic -- rely on --repeats).")
                    _TEMPERATURE_UNSUPPORTED_MODELS.add(self.model)
                    del request_kwargs["temperature"]
                    response = client.messages.create(**request_kwargs)
                else:
                    raise

            # Accumulate across ALL attempts, not just the final one -- a
            # truncated attempt still consumed (and was billed for) real
            # input/output tokens, including the wasted thinking tokens.
            # Undercounting this would make truncation-prone configs look
            # artificially cheap in the cost report, which is exactly
            # backwards for a project about honest cost accounting.
            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens

            text = "".join(block.text for block in response.content if block.type == "text")
            if text.strip():
                break
            block_types = [b.type for b in response.content]
            print(f"  [WARNING] Empty text response from {self.model} "
                  f"(agent_role={self.agent_role}, attempt={attempt+1}/{len(budgets_to_try)}). "
                  f"stop_reason={response.stop_reason}, content_block_types={block_types}, "
                  f"max_tokens={budget}")

        return text, total_input_tokens, total_output_tokens

    # -- Local backend (Ollama-compatible HTTP API) --------------------------
    def _call_local(self, system_prompt: str, user_prompt: str, max_tokens: int):
        # Uses Ollama's /api/generate endpoint. If you're using raw MLX instead,
        # swap this method for a direct mlx_lm.generate() call -- the rest of
        # the agent code doesn't need to change either way.
        payload = {
            "model": "llama3.2:3b",  # override to whatever you've pulled locally
            "prompt": f"{system_prompt}\n\n{user_prompt}",
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                # Deterministic decoding: for structured-extraction tasks we want
                # the same input to reliably produce the same output. Ollama's
                # default temperature (~0.8) is tuned for creative chat, not
                # this kind of consistent-parsing task.
                "temperature": 0.0,
            },
        }
        req = urllib.request.Request(
            self.local_endpoint,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())

        text = data.get("response", "")
        # Ollama reports counts as prompt_eval_count / eval_count.
        in_tok = data.get("prompt_eval_count", 0)
        out_tok = data.get("eval_count", 0)
        return text, in_tok, out_tok
