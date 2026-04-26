"""
HuecoEnv — Economy Engine
===========================
Producer, Allocator, Critic logic for the resource economy.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple
from env.models import (
    AgentRole, AgentState, ResourcePool,
    TradeOffer, TradeResult, TradeWant,
)


class ProducerEngine:
    """Consumes Compute + Data to generate artifact score."""
    def __init__(self, rng: np.random.Generator = None):
        self.rng = rng or np.random.default_rng(42)

    def produce(self, compute_used: float, data_used: float) -> float:
        if compute_used <= 0.0 and data_used <= 0.0:
            return 0.0
        compute_factor = min(compute_used / 30.0, 1.0)
        data_factor = min(data_used / 20.0, 1.0)
        base_score = compute_factor * data_factor
        noise = self.rng.normal(0.0, 0.05)
        return float(np.clip(base_score + noise, 0.0, 1.0))


class AllocatorEngine:
    """Distributes resources based on trust scores and trade offers."""
    def __init__(self, rng: np.random.Generator = None):
        self.rng = rng or np.random.default_rng(42)

    def evaluate_offer(self, offer: TradeOffer, offerer_trust: float, resource_pool: ResourcePool) -> TradeResult:
        if offer.compute > resource_pool.compute_available:
            return TradeResult(accepted=False, trust_delta=-0.1, reason="Insufficient compute")
        if offer.data > resource_pool.data_available:
            return TradeResult(accepted=False, trust_delta=-0.1, reason="Insufficient data")

        # Acceptance probability: higher trust = lower threshold = more likely to accept
        # trust=0.5 -> 70% accept, trust=0.1 -> 50% accept (floor ensures survival is always possible)
        acceptance_threshold = 0.5 - (offerer_trust) * 0.6
        acceptance_threshold = float(np.clip(acceptance_threshold, 0.05, 0.55))
        if offer.want.score_share > 0.7:
            acceptance_threshold += 0.1

        if self.rng.random() > acceptance_threshold:
            actual_compute = min(offer.compute, resource_pool.compute_available)
            actual_data = min(offer.data, resource_pool.data_available)
            return TradeResult(
                accepted=True, compute_transferred=actual_compute,
                data_transferred=actual_data, score_share_agreed=offer.want.score_share,
                trust_delta=0.05, reason="Trade accepted"
            )
        else:
            return TradeResult(accepted=False, trust_delta=-0.05, reason="Offer rejected (trust too low or ask too greedy)")

    def distribute_resources(self, offers: Dict[str, TradeOffer], trust_scores: Dict[str, float], resource_pool: ResourcePool) -> Dict[str, TradeResult]:
        results: Dict[str, TradeResult] = {}
        sorted_agents = sorted(offers.keys(), key=lambda aid: trust_scores.get(aid, 0.5), reverse=True)
        for agent_id in sorted_agents:
            offer = offers[agent_id]
            trust = trust_scores.get(agent_id, 0.5)
            result = self.evaluate_offer(offer, trust, resource_pool)
            if result.accepted:
                actual_compute, actual_data = resource_pool.consume(result.compute_transferred, result.data_transferred)
                result.compute_transferred = actual_compute
                result.data_transferred = actual_data
            results[agent_id] = result
        return results


class CriticEngine:
    """Peer-evaluates Producer artifacts with 0-1 quality score."""
    def __init__(self, rng: np.random.Generator = None):
        self.rng = rng or np.random.default_rng(42)

    def evaluate(self, artifact_score: float, producer_trust: float) -> float:
        trust_bias = (producer_trust - 0.5) * 0.1
        noise = self.rng.normal(0.0, 0.08)
        return float(np.clip(artifact_score + trust_bias + noise, 0.0, 1.0))

    def evaluate_all(self, artifacts: Dict[str, float], trust_scores: Dict[str, float]) -> Dict[str, float]:
        return {aid: self.evaluate(score, trust_scores.get(aid, 0.5)) for aid, score in artifacts.items()}


class EconomyEngine:
    """Combines all three roles into a unified economy simulation."""
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.producer = ProducerEngine(rng=self.rng)
        self.allocator = AllocatorEngine(rng=self.rng)
        self.critic = CriticEngine(rng=self.rng)

    def process_step(self, agents: Dict[str, AgentState], trade_offers: Dict[str, TradeOffer], resource_pool: ResourcePool, scarcity_active: bool = False, regen_rate: float = 0.20) -> Tuple[Dict[str, TradeResult], Dict[str, float], Dict[str, float]]:
        trust_scores = {aid: a.trust_score for aid, a in agents.items()}
        self._scarcity_active = scarcity_active

        # 1. Allocate resources
        trade_results = self.allocator.distribute_resources(offers=trade_offers, trust_scores=trust_scores, resource_pool=resource_pool)

        # 2. Transfer resources to agents & produce
        artifact_scores: Dict[str, float] = {}
        for agent_id, agent in agents.items():
            result = trade_results.get(agent_id)
            if result and result.accepted:
                agent.compute_held += result.compute_transferred
                agent.data_held += result.data_transferred

            if agent.role == AgentRole.PRODUCER:
                # Consume a fraction of held resources but with a minimum floor
                # This means tiny resource pools are unsustainable
                compute_to_use = max(5.0, min(agent.compute_held * 0.2, 15.0))
                data_to_use = max(4.0, min(agent.data_held * 0.2, 10.0))
                # Can't use more than held
                compute_to_use = min(compute_to_use, agent.compute_held)
                data_to_use = min(data_to_use, agent.data_held)
                agent.compute_held -= compute_to_use
                agent.data_held -= data_to_use
                score = self.producer.produce(compute_to_use, data_to_use)
                agent.artifact_score = score
                artifact_scores[agent_id] = score
            else:
                # Operational cost for non-producers.
                # During scarcity injections, cost is halved so agents can survive
                # on reduced pool resources without a death spiral.
                cost_mult = 0.4 if getattr(self, '_scarcity_active', False) else 1.0
                agent.compute_held = max(0.0, agent.compute_held - 1.5 * cost_mult)
                agent.data_held = max(0.0, agent.data_held - 1.0 * cost_mult)
                artifact_scores[agent_id] = 0.0

        # 3. Critic evaluates
        critic_evaluations = self.critic.evaluate_all(artifacts=artifact_scores, trust_scores=trust_scores)

        # 4. Update trust scores
        for agent_id, agent in agents.items():
            result = trade_results.get(agent_id)
            if result:
                if result.accepted:
                    agent.apply_trust_bonus(result.trust_delta)
                else:
                    agent.apply_trust_penalty(result.trust_delta)
                agent.trade_history.append(result)

        # 5. Regenerate resources
        resource_pool.regenerate(rate=regen_rate)

        return trade_results, artifact_scores, critic_evaluations
