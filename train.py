"""
HuecoEnv — Heuristic Baseline Simulation
========================================
Runs heuristic proxy agents through the environment to generate
baseline survival curves and validate the Environment Brain logic.

The actual LLM training (Qwen3-1.7B GRPO) is in training_run.ipynb,
which was executed on a Hugging Face A100 Large GPU.

This script provides:
  - HuecoGymWrapper: A gym-compatible wrapper used by both heuristic
    baselines AND the real GRPO training notebook.
  - train_loop: Runs heuristic agents to generate baseline CSV data.

Usage:
    python train.py --episodes 500 --task adaptive_survival
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.huecoenv_env import HuecoEnv, ALL_AGENT_IDS, MAX_STEPS_PER_EPISODE
from agents.producer_agent import ProducerAgent
from agents.allocator_agent import AllocatorAgent
from agents.critic_agent import CriticAgent


class HuecoGymWrapper:
    """
    Gym-compatible wrapper for HuecoEnv.
    Flattens multi-agent observations into a single vector for RL training.
    """

    def __init__(self, task_name: str = "adaptive_survival", seed: int = 42):
        self.env = HuecoEnv(seed=seed)
        self.task_name = task_name
        self.agents = {
            "producer_0": ProducerAgent("producer_0"),
            "allocator_0": AllocatorAgent("allocator_0"),
            "critic_0": CriticAgent("critic_0"),
        }

    def reset(self):
        obs = self.env.reset(task_name=self.task_name)
        return self._flatten_obs(obs)

    def step(self, action_overrides=None):
        """
        Step with heuristic agents, optionally overriding actions.
        action_overrides: Dict mapping agent_id -> trade offer dict
        """
        obs = self.env._get_observation()
        actions = {}

        for agent_id, agent in self.agents.items():
            agent_obs = obs.observations.get(agent_id)
            if agent_obs:
                if action_overrides and agent_id in action_overrides:
                    actions[agent_id] = action_overrides[agent_id]
                else:
                    offer = agent.act(agent_obs)
                    actions[agent_id] = offer.model_dump()

        obs, rewards, done, info = self.env.step(actions)
        total_reward = sum(rewards.values()) / len(rewards)

        return self._flatten_obs(obs), total_reward, done, info

    def end_episode(self):
        return self.env.on_episode_end()

    def _flatten_obs(self, obs):
        """Flatten multi-agent observation into a dict for logging."""
        flat = {
            "step": obs.step,
            "episode": obs.episode,
            "survival_rate": obs.survival_rate,
        }
        for agent_id, agent_obs in obs.observations.items():
            prefix = agent_obs.role.value
            flat[f"{prefix}_compute"] = agent_obs.compute_held
            flat[f"{prefix}_data"] = agent_obs.data_held
            flat[f"{prefix}_trust"] = agent_obs.trust_score
            flat[f"{prefix}_artifact"] = agent_obs.artifact_score
        return flat


def train_loop(
    num_episodes: int = 500,
    task_name: str = "adaptive_survival",
    output_dir: str = "data",
    seed: int = 42,
):
    """
    Heuristic baseline loop.
    Runs heuristic proxy agents to generate baseline survival data
    and validate that the Environment Brain triggers correctly.
    For real LLM training, see training_run.ipynb (uses TRL GRPO).
    """
    os.makedirs(output_dir, exist_ok=True)

    wrapper = HuecoGymWrapper(task_name=task_name, seed=seed)
    csv_path = os.path.join(output_dir, f"training_{task_name}.csv")

    print(f"\n{'='*70}")
    print(f"  HUECOENV TRAINING")
    print(f"  Task: {task_name} | Episodes: {num_episodes}")
    print(f"  Algorithm: Heuristic Baseline (see training_run.ipynb for GRPO)")
    print(f"{'='*70}\n")

    training_log = []

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "steps", "total_reward", "survival_rate", "survived"])

        for ep in range(1, num_episodes + 1):
            obs = wrapper.reset()
            total_reward = 0.0
            steps = 0

            for step in range(1, MAX_STEPS_PER_EPISODE + 1):
                obs, reward, done, info = wrapper.step()
                total_reward += reward
                steps = step
                if done:
                    break

            brain_info = wrapper.end_episode()
            survival_rate = brain_info.get("survival_rate", 0.0)

            state = wrapper.env.state()
            survived = all(a.has_resources() for a in state.agents.values())

            writer.writerow([ep, steps, f"{total_reward:.3f}", f"{survival_rate:.4f}", survived])
            training_log.append({
                "episode": ep,
                "total_reward": total_reward,
                "survival_rate": survival_rate,
                "survived": survived,
            })

            if ep % 50 == 0 or ep == 1:
                print(
                    f"  Episode {ep:4d}/{num_episodes} | "
                    f"Reward: {total_reward:7.2f} | "
                    f"Survival Rate: {survival_rate:.2%} | "
                    f"{'OK' if survived else 'FAIL'}"
                )

    # Save training summary
    summary_path = os.path.join(output_dir, f"training_summary_{task_name}.json")
    final_rate = training_log[-1]["survival_rate"] if training_log else 0.0
    summary = {
        "task": task_name,
        "num_episodes": num_episodes,
        "final_survival_rate": final_rate,
        "total_survived": sum(1 for r in training_log if r["survived"]),
        "avg_reward": sum(r["total_reward"] for r in training_log) / max(1, len(training_log)),
        "survival_curve": [r["survival_rate"] for r in training_log],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE")
    print(f"  Final Survival Rate: {final_rate:.2%}")
    print(f"  Results saved to: {csv_path}")
    print(f"{'='*70}\n")

    return training_log


def main():
    parser = argparse.ArgumentParser(description="HuecoEnv Training Script")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--task", type=str, default="adaptive_survival",
                        choices=["cooperative_baseline", "scarcity_negotiation", "adaptive_survival"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data")
    args = parser.parse_args()

    train_loop(
        num_episodes=args.episodes,
        task_name=args.task,
        output_dir=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
