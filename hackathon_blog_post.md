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

When the Producer asks for `{"compute": 20, "data": 15}`, the environment's internal Python math engine checks the exact ledger. If the Allocator approves the trade but doesn't have the resources, the transaction mathematically fails. The survival of an agent is tied strictly to a hard-coded mathematical truth: `resources > 0`. This provides a pristine, deterministic reward signal for PPO/GRPO optimization.

### 2. The Environment Brain (Recursive Scarcity)
We built an intelligent `EnvironmentBrain` that monitors the agents' performance. It features a `Sentinel` that calculates a 20-episode rolling average of the multi-agent survival rate.

The moment the RL agents master the environment and achieve an 85% survival rate, the Sentinel triggers the `Injector`. The Injector artificially crashes the global resource capacity by 70% (a "Scarcity Drought").

The agents, which were previously comfortable, suddenly begin to starve. Their previously optimal policies fail. To survive, they must discover entirely new, highly-cooperative policies — such as sacrificing personal trust scores to keep weaker agents alive. 

## The Results

By establishing a deterministic JSON-based reward system and a recursively escalating difficulty curve, HuecoEnv ensures that RL training never flatlines. The agents are forced into a state of continuous adaptation.

In our baseline testing using untrained Llama-3.1-8B models via the Hugging Face Inference API, the models achieved a 99% survival rate during the easy phase. However, the moment the Environment Brain triggered the Scarcity Drought, survival crashed. 

By applying **TRL (Transformer Reinforcement Learning)** and **GRPO**, we are able to train the models to recognize scarcity patterns and adapt their JSON requests dynamically, recovering the survival rate entirely through verifiable, math-based cooperation.

---
*Built for the Meta OpenEnv Hackathon*
