"""
HuecoEnv — Multi-Episode Simulation Runner
==============================================
Runs N episodes with 3 agents, logs survival_rate per episode to CSV,
generates survival curve data, and supports untrained vs trained comparison.

Usage:
    python simulate.py --episodes 50 --agents default
    python simulate.py --episodes 100 --task adaptive_survival
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


def create_default_agents():
    """Create heuristic-based agents for simulation."""
    return {
        "producer_0": ProducerAgent("producer_0"),
        "allocator_0": AllocatorAgent("allocator_0"),
        "critic_0": CriticAgent("critic_0"),
    }


def run_episode(env: HuecoEnv, agents: dict, task_name: str, episode_num: int) -> dict:
    """Run a single episode and return results."""
    obs = env.reset(task_name=task_name)

    total_rewards = {aid: 0.0 for aid in ALL_AGENT_IDS}
    steps_taken = 0

    for step in range(1, MAX_STEPS_PER_EPISODE + 1):
        # Collect actions from all agents
        actions = {}
        for agent_id, agent in agents.items():
            agent_obs = obs.observations.get(agent_id)
            if agent_obs:
                offer = agent.act(agent_obs)
                actions[agent_id] = offer.model_dump()
            else:
                actions[agent_id] = {"compute": 5.0, "data": 3.0, "want": {"score_share": 0.2}}

        # Step environment
        obs, rewards, done, info = env.step(actions)
        steps_taken = step

        for aid, r in rewards.items():
            total_rewards[aid] += r

        if done:
            break

    # End episode and get brain info
    brain_info = env.on_episode_end()
    state = env.state()

    all_survived = all(a.has_resources() for a in state.agents.values())

    return {
        "episode": episode_num,
        "task": task_name,
        "steps_taken": steps_taken,
        "all_survived": all_survived,
        "survival_rate": brain_info.get("survival_rate", 0.0),
        "total_rewards": total_rewards,
        "injection_active": brain_info.get("injection_active", False),
        "injection_level": brain_info.get("injection_level", 0),
        "triggered_injection": brain_info.get("triggered_injection", False),
        "trust_scores": {aid: a.trust_score for aid, a in state.agents.items()},
        "artifact_scores": {aid: a.artifact_score for aid, a in state.agents.items()},
        "compute_available": state.resource_pool.compute_available,
        "data_available": state.resource_pool.data_available,
    }


def run_simulation(
    num_episodes: int = 50,
    task_name: str = "cooperative_baseline",
    output_dir: str = "data",
    seed: int = 42,
) -> list:
    """Run a full simulation and save results to CSV."""
    os.makedirs(output_dir, exist_ok=True)

    env = HuecoEnv(seed=seed)
    agents = create_default_agents()

    results = []
    csv_path = os.path.join(output_dir, f"simulation_{task_name}.csv")

    print(f"\n{'='*70}")
    print(f"  HUECOENV SIMULATION")
    print(f"  Task: {task_name} | Episodes: {num_episodes} | Seed: {seed}")
    print(f"{'='*70}\n")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "steps", "all_survived", "survival_rate",
            "injection_active", "injection_level", "triggered_injection",
            "producer_trust", "allocator_trust", "critic_trust",
            "producer_reward", "allocator_reward", "critic_reward",
            "compute_available", "data_available",
        ])

        for ep in range(1, num_episodes + 1):
            result = run_episode(env, agents, task_name, ep)
            results.append(result)

            writer.writerow([
                result["episode"],
                result["steps_taken"],
                result["all_survived"],
                f"{result['survival_rate']:.4f}",
                result["injection_active"],
                result["injection_level"],
                result["triggered_injection"],
                f"{result['trust_scores'].get('producer_0', 0):.3f}",
                f"{result['trust_scores'].get('allocator_0', 0):.3f}",
                f"{result['trust_scores'].get('critic_0', 0):.3f}",
                f"{result['total_rewards'].get('producer_0', 0):.2f}",
                f"{result['total_rewards'].get('allocator_0', 0):.2f}",
                f"{result['total_rewards'].get('critic_0', 0):.2f}",
                f"{result['compute_available']:.1f}",
                f"{result['data_available']:.1f}",
            ])

            # Print progress
            status = "[OK] SURVIVED" if result["all_survived"] else "[!!] FAILED"
            inject_str = ""
            if result["triggered_injection"]:
                inject_str = f" [INJECTION L{result['injection_level']}!]"
            elif result["injection_active"]:
                inject_str = f" [DROUGHT L{result['injection_level']}]"

            print(
                f"  Episode {ep:3d}/{num_episodes} | {status} | "
                f"Survival Rate: {result['survival_rate']:.2%} | "
                f"Steps: {result['steps_taken']:2d}{inject_str}"
            )

    # Print summary
    final_rate = results[-1]["survival_rate"] if results else 0.0
    total_survived = sum(1 for r in results if r["all_survived"])
    total_injections = sum(1 for r in results if r["triggered_injection"])

    print(f"\n{'='*70}")
    print(f"  SIMULATION COMPLETE")
    print(f"  Final Survival Rate: {final_rate:.2%}")
    print(f"  Episodes Survived: {total_survived}/{num_episodes}")
    print(f"  Total Injections Triggered: {total_injections}")
    print(f"  Results saved to: {csv_path}")
    print(f"  Summary saved to: {os.path.join(output_dir, f'summary_{task_name}.json')}")
    print(f"{'='*70}\n")

    # Save summary JSON
    summary_path = os.path.join(output_dir, f"summary_{task_name}.json")
    summary = {
        "task": task_name,
        "num_episodes": num_episodes,
        "seed": seed,
        "final_survival_rate": final_rate,
        "total_survived": total_survived,
        "total_injections": total_injections,
        "survival_curve": [r["survival_rate"] for r in results],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="HuecoEnv Simulation Runner")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes to run")
    parser.add_argument("--task", type=str, default="cooperative_baseline",
                        choices=["cooperative_baseline", "scarcity_negotiation", "adaptive_survival"],
                        help="Task to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="data", help="Output directory")
    args = parser.parse_args()

    run_simulation(
        num_episodes=args.episodes,
        task_name=args.task,
        output_dir=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
