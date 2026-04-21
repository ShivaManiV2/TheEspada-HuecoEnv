"""
HuecoEnv — LLM Inference Runner
====================================
Runs the 3-agent HuecoEnv environment with LLM-powered agents.
Compatible with the OpenEnv evaluation framework.

Usage:
    python inference.py
"""

import sys
import os
import json
import httpx
from typing import List
from openai import OpenAI

# ── Always resolve imports relative to this file ────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env.huecoenv_env import HuecoEnv, ALL_AGENT_IDS, MAX_STEPS_PER_EPISODE
from env.models import AgentRole
from agents.llm_agent import LLMAgent
# ────────────────────────────────────────────────────────────────────────────

BENCHMARK = "HuecoEnv"
MAX_STEPS = MAX_STEPS_PER_EPISODE
NUM_EPISODES = 10  # Episodes per task for evaluation


def log_start(task: str) -> None:
    print(f"[START] task={task}", flush=True)


def log_step(step: int, reward: float) -> None:
    print(f"[STEP] step={step} reward={reward:.2f}", flush=True)


def log_end(task: str, score: float, steps: int) -> None:
    score_clamped = max(0.01, min(0.99, score))
    print(f"[END] task={task} score={score_clamped:.4f} steps={steps}", flush=True)


def get_client() -> OpenAI:
    """Initialize OpenAI client safely."""
    api_base = os.getenv("API_BASE_URL")
    if not api_base:
        api_base = "https://api.openai.com/v1"

    api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or "dummy-key"

    # explicitly provide httpx.Client to prevent older openai versions
    # from crashing on "proxies" kwarg in modern httpx >= 0.28.
    return OpenAI(base_url=api_base, api_key=api_key, http_client=httpx.Client())


def create_llm_agents(client: OpenAI) -> dict:
    """Create LLM-powered agents for all three roles."""
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    return {
        "producer_0": LLMAgent("producer_0", AgentRole.PRODUCER, client=client, model=model),
        "allocator_0": LLMAgent("allocator_0", AgentRole.ALLOCATOR, client=client, model=model),
        "critic_0": LLMAgent("critic_0", AgentRole.CRITIC, client=client, model=model),
    }


def run_task(client: OpenAI, task_name: str) -> None:
    """Run multiple episodes of a task and report the score."""
    log_start(task=task_name)

    steps_taken: int = 0
    score: float = 0.0
    episode_results = []

    try:
        env = HuecoEnv()
        agents = create_llm_agents(client)

        for episode in range(NUM_EPISODES):
            obs = env.reset(task_name=task_name)
            episode_reward = 0.0
            ep_steps = 0

            for step in range(1, MAX_STEPS + 1):
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
                total_reward = sum(rewards.values()) / len(rewards)
                episode_reward += total_reward
                ep_steps = step
                steps_taken += 1

                log_step(step=steps_taken, reward=total_reward)

                if done:
                    break

            # End episode
            brain_info = env.on_episode_end()
            state = env.state()

            episode_results.append({
                "all_survived": all(a.has_resources() for a in state.agents.values()),
                "steps_taken": ep_steps,
                "survival_rate": brain_info.get("survival_rate", 0.0),
                "episode_reward": episode_reward,
            })

        # Calculate final score = survival rate
        survived_count = sum(1 for r in episode_results if r["all_survived"])
        score = survived_count / len(episode_results) if episode_results else 0.0

    except Exception as exc:
        print(f"[DEBUG] run_task error ({task_name}): {exc}", file=sys.stderr, flush=True)

    finally:
        if steps_taken == 0:
            steps_taken = 1
            log_step(step=1, reward=0.0)

        log_end(task=task_name, score=score, steps=steps_taken)


def main() -> None:
    try:
        client = get_client()

        for task in ("cooperative_baseline", "scarcity_negotiation", "adaptive_survival"):
            run_task(client, task)

    except Exception as exc:
        print(f"[DEBUG] CRITICAL main error: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
