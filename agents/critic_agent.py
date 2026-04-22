"""
HuecoEnv — Critic Agent
=========================
The evaluator: peer-reviews artifacts, minimal resource needs.
"""

from __future__ import annotations
from env.models import AgentRole, AgentObservation, TradeOffer, TradeWant
from agents.base_agent import BaseAgent


class CriticAgent(BaseAgent):
    """Heuristic-based Critic agent."""

    def __init__(self, agent_id: str = "critic_0"):
        super().__init__(agent_id=agent_id, role=AgentRole.CRITIC)
        self.compute_target = 5.0
        self.data_target = 5.0

    def act(self, observation: AgentObservation) -> TradeOffer:
        self.last_observation = observation

        if observation.last_trade:
            if observation.last_trade.accepted:
                self.compute_target = min(8.0, self.compute_target + 0.1)
                self.data_target = min(8.0, self.data_target + 0.1)
            else:
                self.compute_target = max(1.5, self.compute_target * 0.95)
                self.data_target = max(1.5, self.data_target * 0.95)

        score_share = 0.1
        if observation.trust_score < 0.4:
            score_share = 0.05

        return TradeOffer(
            compute=self.compute_target,
            data=self.data_target,
            want=TradeWant(score_share=score_share),
        )
