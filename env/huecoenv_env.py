"""
HuecoEnv — Core Multi-Agent Environment
==========================================
The main environment class. Orchestrates the 3-agent resource economy:
  - 3 Agents: Producer, Allocator, Critic
  - 2 Resources: Compute (100 units), Data (80 units)
  - Economy Engine: Trade protocol + trust + artifact scoring
  - Environment Brain: Sentinel + Injector + World Memory

Compatible with OpenEnv evaluation framework.
"""

from __future__ import annotations

import os
import numpy as np
from typing import Any, Dict, Optional, Tuple

from env.models import (
    AgentRole,
    AgentState,
    HuecoEnvState,
    HuecoObservation,
    AgentObservation,
    MarketConditions,
    ResourcePool,
    TradeOffer,
    TradeWant,
)
from env.economy import EconomyEngine
from env.environment_brain import EnvironmentBrain

# ── Constants ────────────────────────────────────────────────────────────────
ALL_AGENT_IDS = ["producer_0", "allocator_0", "critic_0"]
MAX_STEPS_PER_EPISODE = 50

# Task configs: (compute_cap, data_cap, enable_brain)
TASK_CONFIG = {
    "cooperative_baseline":  (100.0, 80.0, False),
    "scarcity_negotiation":  (60.0,  48.0, False),
    "adaptive_survival":     (100.0, 80.0, True),
    "training_mode":         (100.0, 80.0, True),  # Brutal settings
}

# Starting resources given to each agent at episode start
INITIAL_RESOURCES = {
    AgentRole.PRODUCER:  {"compute": 20.0, "data": 15.0},
    AgentRole.ALLOCATOR: {"compute": 10.0, "data": 5.0},
    AgentRole.CRITIC:    {"compute": 5.0,  "data": 5.0},
}


class HuecoEnv:
    """
    HuecoEnv: Self-Improving Multi-Agent Resource Economy.

    Three agents (Producer, Allocator, Critic) must negotiate two scarce
    resources (Compute and Data) across a series of episodes. An Environment
    Brain monitors survival and injects scarcity droughts to escalate
    difficulty recursively when agents master the current level.
    """

    def __init__(self, seed: int = 42, world_memory_path: str = None):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.world_memory_path = world_memory_path or os.path.join("data", "world_memory.json")

        # Core components
        self.economy = EconomyEngine(seed=seed)
        self.brain = EnvironmentBrain(world_memory_path=self.world_memory_path)

        # Environment state
        self.agents: Dict[str, AgentState] = {}
        self.resource_pool = ResourcePool()
        self.step_count = 0
        self.episode_count = 0
        self.task_name = "cooperative_baseline"
        self._brain_enabled = False
        self._last_rewards: Dict[str, float] = {}

        # Survival tracking
        self._survival_history = []

        self.reset(task_name="cooperative_baseline")

    # ─── Reset ───────────────────────────────────────────────────────────────

    def reset(
        self,
        task_name: str = "cooperative_baseline",
        seed: Optional[int] = None,
    ) -> HuecoObservation:
        """Reset environment for a new episode."""
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
            self.economy = EconomyEngine(seed=seed)

        self.task_name = task_name
        compute_cap, data_cap, brain_enabled = TASK_CONFIG.get(
            task_name, (100.0, 80.0, True)
        )
        self._brain_enabled = brain_enabled
        self.step_count = 0

        # Set up resource pool
        self.resource_pool = ResourcePool(
            compute_capacity=compute_cap,
            data_capacity=data_cap,
            compute_available=compute_cap,
            data_available=data_cap,
        )

        # Apply any active injection from brain (carry over between episodes)
        if self._brain_enabled and self.brain.injector.is_active():
            self.brain.injector.apply(self.resource_pool)

        # Initialise agents
        drought_active = self.brain.injector.is_active() if self._brain_enabled else False
        self.agents = {}
        role_map = {
            "producer_0":  AgentRole.PRODUCER,
            "allocator_0": AgentRole.ALLOCATOR,
            "critic_0":    AgentRole.CRITIC,
        }
        for agent_id, role in role_map.items():
            base = INITIAL_RESOURCES[role]
            if drought_active:
                # During drought, agents start with reduced but survivable resources
                compute_start = base["compute"] * 0.5
                data_start    = base["data"]    * 0.5
            else:
                compute_start = base["compute"]
                data_start    = base["data"]

            self.agents[agent_id] = AgentState(
                agent_id=agent_id,
                role=role,
                compute_held=compute_start,
                data_held=data_start,
            )

        self._last_rewards = {aid: 0.0 for aid in ALL_AGENT_IDS}
        return self._get_observation()

    # ─── Step ────────────────────────────────────────────────────────────────

    def step(
        self,
        actions: Dict[str, Dict[str, Any]],
    ) -> Tuple[HuecoObservation, Dict[str, float], bool, Dict]:
        """
        Execute one environment step.

        Args:
            actions: Dict mapping agent_id -> trade offer dict

        Returns:
            (observation, rewards, done, info)
        """
        self.step_count += 1
        scarcity_on = self._brain_enabled and self.brain.injector.is_active()

        # ── 1. Parse trade offers ──────────────────────────────────────────
        trade_offers: Dict[str, TradeOffer] = {}
        for agent_id in ALL_AGENT_IDS:
            raw = actions.get(agent_id, {})
            try:
                want_raw = raw.get("want", {})
                if isinstance(want_raw, dict):
                    want = TradeWant(**want_raw)
                else:
                    want = TradeWant()
                offer = TradeOffer(
                    compute=float(raw.get("compute", 5.0)),
                    data=float(raw.get("data", 3.0)),
                    want=want,
                )
            except Exception:
                offer = TradeOffer(compute=5.0, data=3.0, want=TradeWant())
            trade_offers[agent_id] = offer

        # ── 2. Passive resource drip ──────────────────────────────────────
        # Dashboard mode gets a massive drip (0.25) so dumb bots survive bad trades
        # and trigger the drought. Training mode gets brutal (0.01) so LLM learns.
        drip_rate = 0.01 if self.task_name == "training_mode" else 0.25
        compute_drip_per_agent = self.resource_pool.compute_capacity * drip_rate / len(ALL_AGENT_IDS)
        data_drip_per_agent   = self.resource_pool.data_capacity    * drip_rate / len(ALL_AGENT_IDS)

        for agent_id, agent in self.agents.items():
            agent.compute_held += compute_drip_per_agent
            agent.data_held    += data_drip_per_agent


        # ── 3. Run economy step ────────────────────────────────────────────
        # Dashboard gets 20% regen, Training gets brutal 10% regen
        regen_rate = 0.10 if self.task_name == "training_mode" else 0.20

        trade_results, artifact_scores, critic_evals = self.economy.process_step(
            agents=self.agents,
            trade_offers=trade_offers,
            resource_pool=self.resource_pool,
            scarcity_active=scarcity_on,
            regen_rate=regen_rate,
        )

        # ── 4. Record strategy for brain ──────────────────────────────────
        if self._brain_enabled:
            accepted_count = sum(1 for r in trade_results.values() if r.accepted)
            avg_compute = sum(o.compute for o in trade_offers.values()) / len(trade_offers)
            avg_data    = sum(o.data    for o in trade_offers.values()) / len(trade_offers)
            avg_share   = sum(o.want.score_share for o in trade_offers.values()) / len(trade_offers)
            self.brain.record_step_strategy(
                avg_compute=avg_compute,
                avg_data=avg_data,
                avg_score_share=avg_share,
                accepted=accepted_count > 0,
            )

        # ── 5. Compute rewards ─────────────────────────────────────────────
        rewards: Dict[str, float] = {}
        for agent_id, agent in self.agents.items():
            r = 0.0
            # Survival bonus/penalty
            if agent.has_resources():
                r += 1.0
            else:
                r -= 2.0

            # Artifact bonus (producers)
            if agent.role == AgentRole.PRODUCER:
                r += artifact_scores.get(agent_id, 0.0) * 2.0

            # Critic eval bonus
            r += critic_evals.get(agent_id, 0.0) * 0.3

            # Trust bonus
            r += agent.trust_score * 0.5

            # Trade outcome
            result = trade_results.get(agent_id)
            if result:
                r += 0.3 if result.accepted else -0.5

            agent.total_reward += r
            rewards[agent_id] = r

        self._last_rewards = rewards

        # ── 6. Check termination ───────────────────────────────────────────
        done = self.step_count >= MAX_STEPS_PER_EPISODE
        all_alive = all(a.has_resources() for a in self.agents.values())
        if not all_alive and self.task_name == "training_mode":
            done = True  # Early end if any agent dies (speeds up training)

        obs = self._get_observation()
        info = {
            "step": self.step_count,
            "all_alive": all_alive,
            "artifact_scores": artifact_scores,
            "critic_evals": critic_evals,
            "brain_state": self.brain.get_state() if self._brain_enabled else {},
        }
        return obs, rewards, done, info

    # ─── Episode end ─────────────────────────────────────────────────────────

    def on_episode_end(self) -> dict:
        """
        Call at the end of every episode.
        Updates brain, survival history, and returns metrics.
        """
        self.episode_count += 1
        all_survived = all(a.has_resources() for a in self.agents.values())
        self._survival_history.append(all_survived)

        if self._brain_enabled:
            brain_info = self.brain.on_episode_end(
                all_survived=all_survived,
                resource_pool=self.resource_pool,
            )
        else:
            brain_info = {
                "injection_active": False,
                "injection_level": 0,
                "triggered_injection": False,
            }

        window = self._survival_history[-20:]
        rate = sum(1 for s in window if s) / max(1, len(window))
        brain_info["survival_rate"] = rate

        return brain_info

    # ─── State + Observation ─────────────────────────────────────────────────

    def _get_observation(self) -> HuecoObservation:
        """Build full multi-agent observation."""
        market = MarketConditions(
            compute_available=self.resource_pool.compute_available,
            data_available=self.resource_pool.data_available,
            compute_capacity=self.resource_pool.compute_capacity,
            data_capacity=self.resource_pool.data_capacity,
            scarcity_active=self.brain.injector.is_active() if self._brain_enabled else False,
            scarcity_episodes_remaining=(
                self.brain.injector.episodes_remaining if self._brain_enabled else 0
            ),
        )
        peer_trust = {
            aid: a.trust_score for aid, a in self.agents.items()
        }

        # Compute rolling survival rate
        window = self._survival_history[-20:]
        sr = sum(1 for s in window if s) / max(1, len(window)) if window else 0.0

        obs_map: Dict[str, AgentObservation] = {}
        for agent_id, agent in self.agents.items():
            obs_map[agent_id] = AgentObservation(
                agent_id=agent_id,
                role=agent.role,
                step=self.step_count,
                episode=self.episode_count,
                compute_held=agent.compute_held,
                data_held=agent.data_held,
                trust_score=agent.trust_score,
                artifact_score=agent.artifact_score,
                peer_trust_scores={k: v for k, v in peer_trust.items() if k != agent_id},
                market=market,
                last_trade=agent.trade_history[-1] if agent.trade_history else None,
            )

        return HuecoObservation(
            observations=obs_map,
            step=self.step_count,
            episode=self.episode_count,
            done=False,
            survival_rate=sr,
        )

    def state(self) -> HuecoEnvState:
        """Return complete environment state (for /state API endpoint)."""
        window = self._survival_history[-20:]
        sr = sum(1 for s in window if s) / max(1, len(window)) if window else 0.0
        return HuecoEnvState(
            episode=self.episode_count,
            step=self.step_count,
            task_name=self.task_name,
            agents=self.agents,
            resource_pool=self.resource_pool,
            world_memory=self.brain.world_memory,
            survival_history=self._survival_history,
            survival_rate=sr,
            scarcity_active=self.brain.injector.is_active() if self._brain_enabled else False,
            scarcity_episodes_remaining=(
                self.brain.injector.episodes_remaining if self._brain_enabled else 0
            ),
            injection_level=(
                self.brain.injector.current_level if self._brain_enabled and self.brain.injector.is_active() else 0
            ),
        )
