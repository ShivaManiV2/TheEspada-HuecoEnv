"""
HuecoEnv — Producer Agent
===========================
The value creator: requests Compute + Data, produces artifacts.
Default strategy: request 40 Compute + 25 Data, offer 50% score share.
"""

from __future__ import annotations
from env.models import AgentRole, AgentObservation, TradeOffer, TradeWant
from agents.base_agent import BaseAgent


class ProducerAgent(BaseAgent):
    """Heuristic-based Producer agent."""

    def __init__(self, agent_id: str = "producer_0"):
        super().__init__(agent_id=agent_id, role=AgentRole.PRODUCER)
        self.compute_target = 35.0
        self.data_target = 25.0

    def act(self, observation: AgentObservation) -> TradeOffer:
        self.last_observation = observation

        # Trial and error learning from last trade (takes ~40 episodes to adapt to severe drought)
        if observation.last_trade:
            if observation.last_trade.accepted:
                self.compute_target = min(45.0, self.compute_target + 1.0)
                self.data_target = min(30.0, self.data_target + 1.0)
            else:
                # Multiply by 0.95: it takes log(0.125)/log(0.95) ~= 40 steps to adapt
                self.compute_target = max(4.0, self.compute_target * 0.95)
                self.data_target = max(3.0, self.data_target * 0.95)

        score_share = 0.5
        # Adjust based on own trust score
        if observation.trust_score < 0.4:
            score_share = 0.3  # Offer more to rebuild trust

        return TradeOffer(
            compute=self.compute_target,
            data=self.data_target,
            want=TradeWant(score_share=score_share),
        )
