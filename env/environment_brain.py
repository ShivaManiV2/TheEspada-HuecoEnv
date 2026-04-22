"""
HuecoEnv — Environment Brain
================================
The recursive self-improvement mechanism:
  - Sentinel:  Monitors rolling 20-episode survival rate window
  - Injector:  Drops resource capacity to create scarcity droughts
  - World Memory: Tracks strategies and triggers recursive escalation
"""

from __future__ import annotations
import json
import os
from typing import List, Optional
from env.models import (
    InjectionLevel, ResourcePool, StrategyFingerprint,
    WorldMemory, WorldMemoryEntry,
)


class Sentinel:
    """
    Watches a rolling window of survival_rate.
    Triggers the Injector if survival_rate >= threshold for window_size consecutive episodes.
    Window is reset after each trigger so agents must re-earn the next injection.
    """
    def __init__(self, window_size: int = 20, threshold: float = 0.85):
        self.window_size = window_size
        self.threshold = threshold
        self.survival_history: List[bool] = []

    def record(self, all_survived: bool):
        """Record whether all agents survived this episode."""
        self.survival_history.append(all_survived)

    def get_survival_rate(self, window: int = None) -> float:
        """Calculate survival rate over the last N episodes."""
        w = window or self.window_size
        if len(self.survival_history) < w:
            recent = self.survival_history
        else:
            recent = self.survival_history[-w:]
        if not recent:
            return 0.0
        return sum(1 for s in recent if s) / len(recent)

    def should_inject(self) -> bool:
        """Check if agents have mastered the current difficulty."""
        if len(self.survival_history) < self.window_size:
            return False
        recent = self.survival_history[-self.window_size:]
        rate = sum(1 for s in recent if s) / len(recent)
        return rate >= self.threshold

    def reset(self):
        """Reset survival history."""
        self.survival_history = []


class Injector:
    """
    Creates scarcity droughts by dropping resource capacity.
    Escalating difficulty levels:
      Level 1: 30% capacity for 10 episodes
      Level 2: 20% capacity for 15 episodes
      Level 3: 20% capacity + one resource disabled for 5 episodes
    """
    def __init__(self):
        self.active = False
        self.current_level = 1
        self.episodes_remaining = 0
        self.capacity_fraction = 1.0
        self.disabled_resource: Optional[str] = None

    def trigger(self, level: int = 1):
        """Activate a scarcity drought at the given level."""
        self.active = True
        self.current_level = level

        if level == 1:
            self.capacity_fraction = 0.30   # 30% capacity — survivable with good trades
            self.episodes_remaining = 10
            self.disabled_resource = None
        elif level == 2:
            self.capacity_fraction = 0.20   # 20% capacity — tight, some agents will die
            self.episodes_remaining = 15
            self.disabled_resource = None
        elif level >= 3:
            self.capacity_fraction = 0.10   # 10% capacity + data disabled — brutal
            self.episodes_remaining = 5
            self.disabled_resource = "data"
        else:
            self.capacity_fraction = 0.30
            self.episodes_remaining = 10
            self.disabled_resource = None

    def apply(self, resource_pool: ResourcePool):
        """Apply current scarcity to the resource pool."""
        if self.active:
            resource_pool.apply_scarcity(self.capacity_fraction, self.disabled_resource)

    def tick(self):
        """Advance one episode. Deactivate if duration expired."""
        if self.active:
            self.episodes_remaining -= 1
            if self.episodes_remaining <= 0:
                self.active = False
                self.capacity_fraction = 1.0
                self.disabled_resource = None

    def is_active(self) -> bool:
        return self.active


class EnvironmentBrain:
    """
    The core innovation: an environment that gets harder as agents get smarter.
    Combines Sentinel monitoring, Injector disruptions, and World Memory tracking.
    """
    def __init__(self, world_memory_path: str = None):
        self.sentinel = Sentinel(window_size=20, threshold=0.85)
        self.injector = Injector()
        self.world_memory = WorldMemory()
        self.world_memory_path = world_memory_path
        self.current_episode = 0

        # Strategy tracking during active injection
        self._injection_compute_allocs: List[float] = []
        self._injection_data_allocs: List[float] = []
        self._injection_score_shares: List[float] = []
        self._injection_acceptances: List[bool] = []
        self._pre_injection_episode: Optional[int] = None

        # Load existing world memory if available
        if world_memory_path and os.path.exists(world_memory_path):
            try:
                with open(world_memory_path, 'r') as f:
                    data = json.load(f)
                    self.world_memory = WorldMemory(**data)
            except (json.JSONDecodeError, Exception):
                pass

    def on_episode_end(self, all_survived: bool, resource_pool: ResourcePool) -> dict:
        """
        Called at the end of every episode.
        Returns info dict about brain state changes.
        """
        self.current_episode += 1
        self.sentinel.record(all_survived)
        info = {
            "survival_rate": self.sentinel.get_survival_rate(),
            "injection_active": self.injector.is_active(),
            "injection_level": self.injector.current_level if self.injector.is_active() else 0,
            "triggered_injection": False,
        }

        # If injection is active, tick it down
        if self.injector.is_active():
            self.injector.tick()
            if not self.injector.is_active():
                # Injection just ended — record recovery and strategy
                self._finalize_injection()
                resource_pool.restore_full_capacity()
                # Reset sentinel so agents must RE-EARN the next injection
                self.sentinel.reset()
                info["injection_ended"] = True
        else:
            # Check if we should trigger a new injection
            if self.sentinel.should_inject():
                level = self._determine_injection_level()
                self.injector.trigger(level)
                self.injector.apply(resource_pool)

                # Reset sentinel IMMEDIATELY after triggering so it
                # cannot fire again until agents survive through the
                # drought AND re-accumulate another full window.
                self.sentinel.reset()

                # Log to world memory
                self.world_memory.log_injection(
                    episode=self.current_episode,
                    level=level,
                    capacity_fraction=self.injector.capacity_fraction,
                    duration=self.injector.episodes_remaining,
                )
                self._pre_injection_episode = self.current_episode
                self._injection_compute_allocs = []
                self._injection_data_allocs = []
                self._injection_score_shares = []
                self._injection_acceptances = []

                info["triggered_injection"] = True
                info["injection_level"] = level

        # If injection is currently active, apply to resource pool
        if self.injector.is_active():
            self.injector.apply(resource_pool)

        # Save world memory
        self._save_world_memory()

        return info

    def record_step_strategy(self, avg_compute: float, avg_data: float, avg_score_share: float, accepted: bool):
        """Record agent strategies during an active injection for fingerprinting."""
        if self.injector.is_active():
            self._injection_compute_allocs.append(avg_compute)
            self._injection_data_allocs.append(avg_data)
            self._injection_score_shares.append(avg_score_share)
            self._injection_acceptances.append(accepted)

    def _finalize_injection(self):
        """Finalize an injection: compute strategy fingerprint and recovery time."""
        entry = self.world_memory.get_last_entry()
        if entry is None:
            return

        # Compute strategy fingerprint
        if self._injection_compute_allocs:
            fp = StrategyFingerprint(
                avg_compute_allocation=sum(self._injection_compute_allocs) / len(self._injection_compute_allocs),
                avg_data_allocation=sum(self._injection_data_allocs) / len(self._injection_data_allocs),
                avg_score_share_requested=sum(self._injection_score_shares) / len(self._injection_score_shares),
                cooperation_ratio=sum(1 for a in self._injection_acceptances if a) / max(1, len(self._injection_acceptances)),
            )
            entry.strategy_fingerprint = fp

        # Calculate recovery time
        if self._pre_injection_episode is not None:
            entry.recovery_time = self.current_episode - self._pre_injection_episode

        entry.resolved = True

    def _determine_injection_level(self) -> int:
        """Determine injection level based on World Memory."""
        # Check if agents reused strategies
        if self._injection_compute_allocs:
            current_fp = StrategyFingerprint(
                avg_compute_allocation=sum(self._injection_compute_allocs) / max(1, len(self._injection_compute_allocs)),
                avg_data_allocation=sum(self._injection_data_allocs) / max(1, len(self._injection_data_allocs)),
                avg_score_share_requested=sum(self._injection_score_shares) / max(1, len(self._injection_score_shares)),
                cooperation_ratio=sum(1 for a in self._injection_acceptances if a) / max(1, len(self._injection_acceptances)),
            )
            if self.world_memory.check_strategy_reuse(current_fp):
                return min(3, self.world_memory.current_injection_level + 1)

        # Default: escalate by 1 each time, cap at 3
        level = min(3, self.world_memory.total_injections + 1)
        self.world_memory.current_injection_level = level
        return level

    def _save_world_memory(self):
        """Persist world memory to disk."""
        if self.world_memory_path:
            try:
                os.makedirs(os.path.dirname(self.world_memory_path), exist_ok=True)
                with open(self.world_memory_path, 'w') as f:
                    json.dump(self.world_memory.model_dump(), f, indent=2)
            except Exception:
                pass

    def get_state(self) -> dict:
        """Return current brain state for observation."""
        return {
            "survival_rate": self.sentinel.get_survival_rate(),
            "injection_active": self.injector.is_active(),
            "injection_level": self.injector.current_level if self.injector.is_active() else 0,
            "episodes_remaining": self.injector.episodes_remaining,
            "total_injections": self.world_memory.total_injections,
            "current_episode": self.current_episode,
        }

    def reset(self):
        """Reset for a new simulation run (keeps world memory)."""
        self.sentinel.reset()
        self.injector = Injector()
        self.current_episode = 0
