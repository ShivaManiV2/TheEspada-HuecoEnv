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

    def act(self, observation: AgentObservation) -> TradeOffer:
        self.last_observation = observation

        # Critic needs minimal resources (just enough to stay alive)
        if observation.market.scarcity_active:
            compute_req = 4.0
            data_req = 3.0
        else:
            compute_req = max(5.0, observation.market.compute_available * 0.08)
            data_req = max(4.0, observation.market.data_available * 0.08)

        # Critic wants small score share — focused on evaluation role
        return TradeOffer(
            compute=compute_req,
            data=data_req,
            want=TradeWant(score_share=0.1),
        )
