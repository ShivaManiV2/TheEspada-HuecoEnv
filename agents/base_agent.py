"""
HuecoEnv — Base Agent
========================
Abstract base class for all HuecoEnv agents.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from env.models import AgentRole, AgentObservation, TradeOffer


class BaseAgent(ABC):
    """Abstract base class for HuecoEnv agents."""

    def __init__(self, agent_id: str, role: AgentRole):
        self.agent_id = agent_id
        self.role = role
        self.last_observation: Optional[AgentObservation] = None

    @abstractmethod
    def act(self, observation: AgentObservation) -> TradeOffer:
        """Generate a trade offer from the current observation."""
        ...

    def update(self, reward: float, info: Dict[str, Any]) -> None:
        """Process reward signal (for learning agents)."""
        pass

    def reset(self) -> None:
        """Reset agent state for a new episode."""
        self.last_observation = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, role={self.role.value})"
