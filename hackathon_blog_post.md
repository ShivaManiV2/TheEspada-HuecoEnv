---
title: "HuecoEnv: Solving the Reward Hacking Problem with Recursive Scarcity"
emoji: 🌍
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# HuecoEnv: Solving the Reward Hacking Problem with Recursive Scarcity

Reinforcement Learning for Large Language Models (LLMs) has historically struggled with a massive bottleneck: **Objective Verifiability**. 

When training LLMs to act as autonomous agents, developers often rely on "LLM-as-a-Judge" to evaluate the agent's actions. This introduces extreme bias, hallucinations, and makes the reward signal subjective. Furthermore, standard environments are static. Once an agent learns the optimal policy, it stops learning. It becomes comfortable.

At the Meta OpenEnv Hackathon, our team built **HuecoEnv** — a multi-agent economic simulation designed to completely eliminate subjective rewards and force continuous curriculum learning through systemic scarcity.

## The Problem: Subjective Rewards and Static Environments

In a standard multi-agent LLM setup, agents might negotiate over resources. If you ask a Critic LLM, *"Did the Producer make a good trade?"*, the Critic might arbitrarily say *Yes* simply because the Producer used polite language. This leads to **Reward Hacking** — the agents learn to manipulate the Critic rather than solving the actual economic constraints.

Additionally, if the environment always provides 100 units of Compute and 100 units of Data, the agents eventually memorize the exact numerical requests that maximize their score. The learning curve flatlines.

## The Solution: HuecoEnv

HuecoEnv operates on two core architectural pillars that solve these problems:

### 1. RLVR (Reinforcement Learning with Verifiable Rewards)
HuecoEnv forces all agents to communicate exclusively through strict JSON contracts (using Pydantic models). There is no "LLM Critic" determining the success of a trade. 

When the Producer asks for `{"compute": 20, "data": 15}`, the environment's internal Python math engine checks the exact ledger. If the Allocator approves the trade but doesn't have the resources, the transaction mathematically fails. The survival of an agent is tied strictly to a hard-coded mathematical truth: `resources > 0`. This provides a pristine, deterministic reward signal for GRPO optimization.

If the LLM produces invalid JSON, it receives a "poison offer" — zero resources, guaranteed rejection. There is no heuristic safety net. The model must learn the protocol or starve.

### 2. The Environment Brain (Recursive Scarcity)
We built an intelligent `EnvironmentBrain` that monitors the agents' performance. It features a `Sentinel` that calculates a 20-episode rolling average of the multi-agent survival rate.

The moment the RL agents master the environment and achieve an 85% survival rate, the Sentinel triggers the `Injector`. The Injector artificially crashes the global resource capacity by up to 90% (a "Scarcity Drought").

The agents, which were previously comfortable, suddenly begin to starve. Their previously optimal policies fail. To survive, they must discover entirely new, highly-cooperative policies — such as sacrificing personal trust scores to keep weaker agents alive.

## The Results

We trained **Qwen3-1.7B** using TRL's GRPO on a Hugging Face A100 GPU across 1,000 episodes. The LLM directly controls the Producer agent, while the Allocator and Critic remain heuristic-driven.

The training curve reveals a distinctive **sawtooth pattern**:

1. **Episodes 1–100 (Struggling):** Survival hovers at 35–45% as the model learns valid JSON formatting and reasonable trade values.
2. **Episodes 400–480 (Mastery):** The model masters the trade protocol. Survival climbs to **75–85%**. The Sentinel detects this mastery and triggers the first Scarcity Drought.
3. **Episodes 480–800 (Collapse & Recovery):** Survival crashes to ~50% as resources are slashed. Over the next 200 episodes, the model discovers more conservative strategies and climbs back to **75–80%**.
4. **Episodes 980–1000 (Second Drought):** The Environment Brain escalates again. Survival drops to ~55%, proving the recursive mechanism works.

This sawtooth pattern — mastery followed by forced collapse followed by adaptation — is exactly the continuous learning signal that static benchmarks cannot provide.

---
*Built for the Meta OpenEnv Hackathon. [GitHub](https://github.com/ShivaManiV2/HuecoEnv)*
