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

    def act(self, observation: AgentObservation) -> TradeOffer:
        self.last_observation = observation

        # Adaptive strategy based on scarcity
        if observation.market.scarcity_active:
            # During drought: request less, offer more share
            compute_req = min(15.0, observation.market.compute_available * 0.4)
            data_req = min(10.0, observation.market.data_available * 0.4)
            score_share = 0.3  # Give more to allocator to maintain trust
        else:
            # Normal: request optimal amounts
            compute_req = min(40.0, observation.market.compute_available * 0.5)
            data_req = min(25.0, observation.market.data_available * 0.5)
            score_share = 0.5

        # Adjust based on own trust score
        if observation.trust_score < 0.3:
            score_share = max(0.2, score_share - 0.15)  # Offer more to rebuild trust

        return TradeOffer(
            compute=compute_req,
            data=data_req,
            want=TradeWant(score_share=score_share),
        )
