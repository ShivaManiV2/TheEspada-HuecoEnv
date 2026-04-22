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
        self.compute_target = 8.0
        self.data_target = 6.0

    def act(self, observation: AgentObservation) -> TradeOffer:
        self.last_observation = observation

        if observation.last_trade:
            if observation.last_trade.accepted:
                self.compute_target = min(12.0, self.compute_target + 0.2)
                self.data_target = min(8.0, self.data_target + 0.2)
            else:
                self.compute_target = max(2.0, self.compute_target * 0.95)
                self.data_target = max(1.5, self.data_target * 0.95)

        score_share = 0.25
        # Allocator is conservative — keeps trust high
        if observation.trust_score < 0.4:
            score_share = 0.15

        return TradeOffer(
            compute=self.compute_target,
            data=self.data_target,
            want=TradeWant(score_share=score_share),
        )
