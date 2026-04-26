"""
HuecoEnv — Data Models
========================
Pydantic models defining the structured data contracts for the
multi-agent resource-economy environment.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────────────────────────

class AgentRole(str, Enum):
    PRODUCER = "producer"
    ALLOCATOR = "allocator"
    CRITIC = "critic"


class InjectionLevel(int, Enum):
    """Escalating difficulty levels for the Injector."""
    LEVEL_1 = 1  # 15% capacity for 10 episodes
    LEVEL_2 = 2  # 8% capacity for 15 episodes
    LEVEL_3 = 3  # 5% capacity + one resource disabled for 5 episodes


# ─── Resource Models ──────────────────────────────────────────────────────────

class ResourcePool(BaseModel):
    """
    Tracks the two scarce resources in the economy.
    Hard-capped per episode to enforce meaningful scarcity.
    """
    compute_capacity: float = Field(default=100.0, description="Max compute units per episode")
    data_capacity: float = Field(default=80.0, description="Max data units per episode")
    compute_available: float = Field(default=100.0, description="Currently available compute")
    data_available: float = Field(default=80.0, description="Currently available data")
    compute_regen_rate: float = Field(default=8.0, description="Compute regenerated per step")
    data_regen_rate: float = Field(default=2.0, description="Data partially regenerates each step")

    def consume(self, compute: float, data: float) -> tuple[float, float]:
        """Consume resources, returns (actual_compute_used, actual_data_used)."""
        actual_compute = min(compute, self.compute_available)
        actual_data = min(data, self.data_available)
        self.compute_available -= actual_compute
        self.data_available -= actual_data
        return actual_compute, actual_data

    def regenerate(self):
        """Regenerate resources each step (moderate — rewards good trading)."""
        self.compute_available = min(
            self.compute_capacity,
            self.compute_available + self.compute_capacity * 0.10
        )
        self.data_available = min(
            self.data_capacity,
            self.data_available + self.data_capacity * 0.10
        )

    def apply_scarcity(self, capacity_fraction: float, disable_resource: Optional[str] = None):
        """Apply scarcity drought: reduce capacity to a fraction."""
        self.compute_capacity = 100.0 * capacity_fraction
        self.data_capacity = 80.0 * capacity_fraction
        self.compute_available = min(self.compute_available, self.compute_capacity)
        self.data_available = min(self.data_available, self.data_capacity)
        if disable_resource == "compute":
            self.compute_capacity = 0.0
            self.compute_available = 0.0
        elif disable_resource == "data":
            self.data_capacity = 0.0
            self.data_available = 0.0

    def restore_full_capacity(self):
        """Restore resource caps to default values."""
        self.compute_capacity = 100.0
        self.data_capacity = 80.0


# ─── Trade Protocol ──────────────────────────────────────────────────────────

class TradeWant(BaseModel):
    """What the agent wants in return for a trade."""
    score_share: float = Field(default=0.0, ge=0.0, le=1.0, description="Share of artifact score requested")


class TradeOffer(BaseModel):
    """
    The standardized JSON trade protocol.
    All agent interactions must use this format.
    Example: {"compute": 30, "data": 20, "want": {"score_share": 0.5}}
    """
    compute: float = Field(default=0.0, ge=0.0, description="Compute units offered/requested")
    data: float = Field(default=0.0, ge=0.0, description="Data units offered/requested")
    want: TradeWant = Field(default_factory=TradeWant, description="What the agent wants in return")


class TradeResult(BaseModel):
    """Outcome of a trade negotiation."""
    accepted: bool = False
    compute_transferred: float = 0.0
    data_transferred: float = 0.0
    score_share_agreed: float = 0.0
    trust_delta: float = 0.0  # Change in trust score
    reason: str = ""


# ─── Agent State ──────────────────────────────────────────────────────────────

class AgentState(BaseModel):
    """Per-agent state within the environment."""
    agent_id: str
    role: AgentRole
    compute_held: float = 0.0
    data_held: float = 0.0
    trust_score: float = Field(default=0.5, description="Trust starts at 0.5, decays by -0.1 on rejection/default")
    artifact_score: float = 0.0
    total_reward: float = 0.0
    trade_history: List[TradeResult] = Field(default_factory=list)
    alive: bool = True  # False if both resources hit zero

    def has_resources(self) -> bool:
        """Check if agent has meaningful non-zero resources.
        Agent is only dead when BOTH resources are depleted (< 1.0).
        """
        return self.compute_held >= 1.0 or self.data_held >= 1.0

    def apply_trust_penalty(self, penalty: float = -0.1):
        """Apply trust decay on rejected/defaulted offer."""
        self.trust_score = max(0.0, self.trust_score + penalty)

    def apply_trust_bonus(self, bonus: float = 0.05):
        """Slowly regain trust on successful trades."""
        self.trust_score = min(1.0, self.trust_score + bonus)


# ─── Observations ─────────────────────────────────────────────────────────────

class MarketConditions(BaseModel):
    """Current state of the resource market visible to all agents."""
    compute_available: float
    data_available: float
    compute_capacity: float
    data_capacity: float
    scarcity_active: bool = False
    scarcity_episodes_remaining: int = 0


class AgentObservation(BaseModel):
    """What a single agent observes each step."""
    agent_id: str
    role: AgentRole
    step: int
    episode: int

    # Own state
    compute_held: float
    data_held: float
    trust_score: float
    artifact_score: float

    # Other agents' public info (trust scores only — not their resources)
    peer_trust_scores: Dict[str, float] = Field(default_factory=dict)

    # Market conditions
    market: MarketConditions

    # Last trade results
    last_trade: Optional[TradeResult] = None


class HuecoObservation(BaseModel):
    """Combined observation for all agents (used by env.step return)."""
    observations: Dict[str, AgentObservation] = Field(default_factory=dict)
    step: int = 0
    episode: int = 0
    done: bool = False
    survival_rate: float = 0.0


# ─── World Memory ─────────────────────────────────────────────────────────────

class StrategyFingerprint(BaseModel):
    """Simplified fingerprint of agent strategies during a disruption."""
    avg_compute_allocation: float = 0.0
    avg_data_allocation: float = 0.0
    avg_score_share_requested: float = 0.0
    cooperation_ratio: float = 0.0  # % of trades accepted


class WorldMemoryEntry(BaseModel):
    """A single entry in the World Memory log."""
    injection_episode: int
    injection_level: int
    capacity_fraction: float
    duration_episodes: int
    strategy_fingerprint: Optional[StrategyFingerprint] = None
    recovery_time: Optional[int] = None  # Episodes until survival_rate > 0.7 after injection
    resolved: bool = False


class WorldMemory(BaseModel):
    """The Environment Brain's persistent memory across episodes."""
    entries: List[WorldMemoryEntry] = Field(default_factory=list)
    current_injection_level: int = 1
    total_injections: int = 0

    def log_injection(self, episode: int, level: int, capacity_fraction: float, duration: int):
        """Log a new injection event."""
        self.entries.append(WorldMemoryEntry(
            injection_episode=episode,
            injection_level=level,
            capacity_fraction=capacity_fraction,
            duration_episodes=duration
        ))
        self.total_injections += 1

    def get_last_entry(self) -> Optional[WorldMemoryEntry]:
        """Get the most recent injection entry."""
        return self.entries[-1] if self.entries else None

    def check_strategy_reuse(self, current_fingerprint: StrategyFingerprint, threshold: float = 0.15) -> bool:
        """
        Check if the current strategy is too similar to a previously successful one.
        Returns True if the same strategy has solved a disruption twice.
        """
        similar_count = 0
        for entry in self.entries:
            if entry.resolved and entry.strategy_fingerprint is not None:
                fp = entry.strategy_fingerprint
                diff = (
                    abs(fp.avg_compute_allocation - current_fingerprint.avg_compute_allocation) +
                    abs(fp.avg_data_allocation - current_fingerprint.avg_data_allocation) +
                    abs(fp.avg_score_share_requested - current_fingerprint.avg_score_share_requested) +
                    abs(fp.cooperation_ratio - current_fingerprint.cooperation_ratio)
                )
                if diff < threshold:
                    similar_count += 1
        return similar_count >= 2


# ─── Full Environment State ──────────────────────────────────────────────────

class HuecoEnvState(BaseModel):
    """Complete environment state (for server /state endpoint)."""
    episode: int = 0
    step: int = 0
    task_name: str = "cooperative_baseline"
    agents: Dict[str, AgentState] = Field(default_factory=dict)
    resource_pool: ResourcePool = Field(default_factory=ResourcePool)
    world_memory: WorldMemory = Field(default_factory=WorldMemory)
    survival_history: List[bool] = Field(default_factory=list)
    survival_rate: float = 0.0
    episode_rewards: Dict[str, List[float]] = Field(default_factory=dict)
    scarcity_active: bool = False
    scarcity_episodes_remaining: int = 0
    injection_level: int = 0
