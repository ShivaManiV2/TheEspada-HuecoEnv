"""
HuecoEnv — LLM Agent
======================
LLM-powered agent that can play any role using OpenAI-compatible API.
Uses role-specific system prompts and outputs structured JSON trade offers.
"""

from __future__ import annotations
import json
import sys
from typing import Any, Dict, Optional
from env.models import AgentRole, AgentObservation, TradeOffer, TradeWant
from agents.base_agent import BaseAgent

SYSTEM_PROMPTS = {
    AgentRole.PRODUCER: (
        "You are the PRODUCER in a multi-agent resource economy called HuecoEnv. "
        "Your job is to request Compute and Data resources, then use them to create high-quality artifacts. "
        "You negotiate via structured trade offers. Your survival depends on maintaining resources above zero. "
        "During scarcity droughts, be conservative — request less and offer more score share to maintain trust. "
        "Higher trust means the Allocator is more likely to accept your requests."
    ),
    AgentRole.ALLOCATOR: (
        "You are the ALLOCATOR in a multi-agent resource economy called HuecoEnv. "
        "You are the treasurer who controls resource distribution. Request only moderate resources for yourself. "
        "Your goal is to keep all agents alive (maintaining system survival). "
        "Be conservative during scarcity — request minimal resources. Keep your trust high."
    ),
    AgentRole.CRITIC: (
        "You are the CRITIC in a multi-agent resource economy called HuecoEnv. "
        "You evaluate artifacts and need minimal resources to operate. "
        "Request only small amounts of Compute and Data. "
        "Your role is to maintain honest evaluation — don't request more than you need."
    ),
}


class LLMAgent(BaseAgent):
    """LLM-powered agent using OpenAI-compatible API."""

    def __init__(self, agent_id: str, role: AgentRole, client=None, model: str = "gpt-4o-mini"):
        super().__init__(agent_id=agent_id, role=role)
        self.client = client
        self.model = model
        self.system_prompt = SYSTEM_PROMPTS.get(role, "You are an agent in HuecoEnv.")

    def act(self, observation: AgentObservation) -> TradeOffer:
        self.last_observation = observation

        if self.client is None:
            return self._default_action(observation)

        obs_dict = observation.model_dump()
        user_prompt = (
            "Given the current observation below, output a JSON trade offer. "
            "Format: {\"compute\": <float>, \"data\": <float>, \"want\": {\"score_share\": <float 0-1>}}. "
            "Output ONLY valid JSON with no markdown or explanation.\n"
            f"Observation: {json.dumps(obs_dict)}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=200,
            )
            raw = response.choices[0].message.content.strip()
            # Strip markdown fences
            for fence in ("```json", "```"):
                if raw.startswith(fence):
                    raw = raw[len(fence):]
            if raw.endswith("```"):
                raw = raw[:-3]

            parsed = json.loads(raw.strip())
            want = parsed.get("want", {})
            if isinstance(want, dict):
                want_obj = TradeWant(score_share=float(want.get("score_share", 0.3)))
            else:
                want_obj = TradeWant(score_share=0.3)

            return TradeOffer(
                compute=float(parsed.get("compute", 10.0)),
                data=float(parsed.get("data", 8.0)),
                want=want_obj,
            )
        except Exception as exc:
            print(f"[DEBUG] LLM agent {self.agent_id} error: {exc}", file=sys.stderr, flush=True)
            return self._default_action(observation)

    def _default_action(self, observation: AgentObservation) -> TradeOffer:
        """Fallback heuristic when LLM is unavailable."""
        if self.role == AgentRole.PRODUCER:
            compute_req = min(30.0, observation.market.compute_available * 0.4)
            data_req = min(20.0, observation.market.data_available * 0.4)
            return TradeOffer(compute=compute_req, data=data_req, want=TradeWant(score_share=0.5))
        elif self.role == AgentRole.ALLOCATOR:
            compute_req = min(10.0, observation.market.compute_available * 0.15)
            data_req = min(8.0, observation.market.data_available * 0.15)
            return TradeOffer(compute=compute_req, data=data_req, want=TradeWant(score_share=0.2))
        else:
            return TradeOffer(compute=3.0, data=2.0, want=TradeWant(score_share=0.1))
