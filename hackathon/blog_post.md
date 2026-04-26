# We built an environment that gets harder as agents get smarter

*By the HuecoEnv Team — OpenEnv Hackathon*

For the past year, the AI research community has been locked in a benchmark arms race. Every time a new evaluation drops, models "solve" it in a matter of months, forcing researchers to invent increasingly convoluted tests just to differentiate between state-of-the-art models. 

But what if the problem isn't the models? What if the problem is the environments? 

Static environments are fundamentally flawed. They assume a fixed level of difficulty. Once an agent learns the optimal policy—whether through PPO, GRPO, or purely in-context learning—the challenge vanishes. The environment ceases to be a proving ground and becomes merely a checklist.

For the OpenEnv hackathon, we decided to change the paradigm. Instead of building a harder benchmark, we built an environment that *gets harder automatically as agents get smarter*. 

We call it **HuecoEnv**.

### The Economy of Survival

HuecoEnv is a multi-agent resource-management environment built on the OpenEnv framework. It strips away domain-specific complexity in favor of pure economic survival.

There are three agents:
1. **The Producer:** Consumes Compute and Data to generate high-quality artifacts. Controlled by the LLM being trained.
2. **The Allocator:** Acts as the treasurer, evaluating trade offers and distributing resources.
3. **The Critic:** Provides robust peer-evaluation of the Producer's artifacts.

These agents communicate using a strict, programmable JSON trade protocol (`{"compute": X, "data": Y, "want": {"score_share": Z}}`). 

There is no complex portfolio math or arbitrary scoring functions. The evaluation mechanism is entirely objective: **Survival Rate**. Can all three agents negotiate trust and resources well enough to avoid zeroing out over a 50-step episode?

### The Environment Brain: The Real Protagonist

What makes HuecoEnv unique is not the agents, but the world they inhabit. We gave the environment a brain.

We implemented an auto-escalating difficulty system that tracks agent performance. The system consists of three parts:
* **The Sentinel:** Watches a rolling 20-episode window. If the `survival_rate` stays above 85% for 20 consecutive episodes, it knows the agents have mastered the current difficulty. 
* **The Injector:** When triggered by the Sentinel, the Injector forces a "Scarcity Drought," slashing the total available compute and data pool to as low as 10% of its normal capacity.
* **World Memory:** A recursive log that tracks strategies. If agents survive a drought using the same strategy twice, the environment learns and makes the next drought even harsher (e.g., entirely disabling a specific resource type).

### The Four Protocols (Task Modes)

To prove this architecture, HuecoEnv ships with distinct simulation protocols:
1. **Protocol Alpha (Cooperative Baseline):** 100% resources, Brain disabled. Proves the LLM can learn JSON formatting without dying.
2. **Protocol Beta (Scarcity Negotiation):** 60% resources, Brain disabled. Forces cutthroat economics from episode one.
3. **Protocol Omega (Adaptive Dashboard):** 100% resources, Brain enabled, relaxed physics. Designed for the interactive UI so observers can visually witness the Scarcity Drought trigger.
4. **A100 Training Mode:** 100% resources, Brain enabled, brutal physics. Passive resource regeneration is slashed to 1%. This is the crucible where the actual LLM was trained.

### Real-World Applications

HuecoEnv isn't just a game; it is a simulation of real-world multi-agent resource constraints:
- **Cloud Compute Arbitration:** AI agents dynamically allocating GPU clusters and data bandwidth in multi-tenant environments (like AWS or Azure) during peak demand surges.
- **Supply Chain Logistics:** Autonomous nodes negotiating scarce physical resources during unexpected global shortages (simulated by the Scarcity Drought).
- **Automated Throttling Systems:** Testing how API rate limiters interact with high-frequency trading algorithms during market volatility.
- **Robust RLHF Benchmarking:** Evaluating if a new foundation model will hack its reward function, or if it can adapt to shifting constraints over a long context window.

### GRPO Training Results

We trained **Qwen3-1.7B** using TRL's Group Relative Policy Optimization (GRPO) on a Hugging Face A100 GPU across 1,000 episodes. The LLM directly controls the Producer agent, generating JSON trade offers that are parsed and fed into the environment. If the model produces invalid JSON, it receives a "poison offer" that guarantees rejection and starvation — there is no heuristic safety net.

The results tell a compelling three-act story:

1. **Act 1 — Struggling (Episodes 1–100):** The untrained model produces mostly invalid JSON. Survival hovers around 35–45% as the model slowly learns the trade protocol structure.

2. **Act 2 — Mastery (Episodes 400–480):** The model has learned to produce valid, well-calibrated trade offers. Survival climbs to 75–85%. The Sentinel detects this mastery and triggers the first Scarcity Drought.

3. **Act 3 — Collapse & Recovery (Episodes 480–800):** Survival crashes back to 45–55% as the resource pool is slashed. But unlike static benchmarks, the model doesn't plateau — it adapts. Over the next 200 episodes, it discovers more conservative trade strategies and climbs back to 75–80%.

This cycle of mastery → collapse → recovery repeats throughout the training run, producing a distinctive "sawtooth" pattern that proves the Environment Brain is working as designed.

### Why This Matters

HuecoEnv proves that the next generation of benchmarks shouldn't be static questionnaires or single-player puzzles. They should be adversarial, adaptive economies that push agents out of their comfort zones dynamically. 

When you build an environment that fights back, you don't just test what an agent knows — you test how quickly it can adapt when everything goes wrong.

*Built for the Meta OpenEnv Hackathon. Check out the [open-source code on GitHub](https://github.com/ShivaManiV2/HuecoEnv).*
