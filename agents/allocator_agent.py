"""
HuecoEnv — Allocator Agent
============================
The treasurer/negotiator: distributes resources based on trust.
Default strategy: equal split weighted by trust scores.
"""

from __future__ import annotations
from env.models import AgentRole, AgentObservation, TradeOffer, TradeWant
from agents.base_agent import BaseAgent


class AllocatorAgent(BaseAgent):
    """Heuristic-based Allocator agent."""

    def __init__(self, agent_id: str = "allocator_0"):
        super().__init__(agent_id=agent_id, role=AgentRole.ALLOCATOR)

    def act(self, observation: AgentObservation) -> TradeOffer:
        self.last_observation = observation

        # Allocator requests moderate resources for operational needs
        if observation.market.scarcity_active:
            compute_req = 5.0
            data_req = 4.0
            score_share = 0.2
        else:
            compute_req = max(8.0, observation.market.compute_available * 0.2)
            data_req = max(6.0, observation.market.data_available * 0.2)
            score_share = 0.25

        # Allocator is conservative — keeps trust high
        if observation.trust_score < 0.4:
            compute_req *= 0.5
            data_req *= 0.5
            score_share = 0.15

        return TradeOffer(
            compute=compute_req,
            data=data_req,
            want=TradeWant(score_share=score_share),
        )
