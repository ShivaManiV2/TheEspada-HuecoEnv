# HuecoEnv
**"We built an environment that gets harder as agents get smarter, automatically."**

---

## Slide 1: The Problem with Benchmarks
**Static Environments Are Dead.**
- Current benchmarks are static puzzles. Once an agent learns the optimal policy, the benchmark becomes useless.
- There is no mechanism for environments to **co-evolve** with agent intelligence.
- **The Result:** We spend more time building new tests than pushing the boundaries of adaptation.

---

## Slide 2: The Solution - HuecoEnv
**An adversarial, self-improving multi-agent economy.**
- We built a 3-agent resource economy (Producer, Allocator, Critic) using the OpenEnv framework.
- The LLM controls the Producer — generating JSON trade offers parsed and executed by the environment.
- Agents must negotiate scarce resources (Compute & Data) using a strict JSON protocol.
- **The Metric:** Survival Rate. (Can all 3 agents cooperate to keep everyone alive for 50 steps?)

---

## Slide 3: The Architecture - The Environment Brain
**The environment is the protagonist.**
- **The Sentinel:** Monitors the 20-episode rolling survival rate. If agents master the environment (>85% survival), it fires a trigger.
- **The Injector:** Introduces "Scarcity Droughts", slashing the resource pool to 10% capacity.
- **World Memory:** The environment remembers agent strategies. If an agent solves a drought the same way twice, the next drought gets recursively harder.

---

## Slide 4: Real Results — Qwen3-1.7B GRPO Training

*(Visual: Survival plot from `assets/survival_plot.png`)*

**Trained on Hugging Face A100 GPU | 1,000 episodes | TRL GRPO**

- **Act 1 — Struggling (Ep 1–100):** Survival at ~35–45%. The LLM is learning the JSON trade protocol.
- **Act 2 — Mastery (Ep 400–480):** Survival climbs to **75–85%**. The Sentinel detects mastery and fires the Injector.
- **Act 3 — Collapse & Recovery (Ep 480–800):** Survival crashes to ~50%, then the model adapts and recovers to **75–80%**.
- **The Sawtooth:** This collapse–recovery cycle repeats, proving the Environment Brain works as designed.

**Key stat:** Peak survival rate of **85%**, with the environment autonomously triggering 2+ scarcity droughts.

---

## Slide 5: The Closing Line
"When you build an environment that fights back, you don't just test what an agent knows — you test how quickly it can adapt when everything goes wrong."

**HuecoEnv — Ready for OpenEnv.**
