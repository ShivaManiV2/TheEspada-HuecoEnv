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
- Agents must negotiate scarce resources (Compute & Data) using a strict JSON trade protocol.
- **The Metric:** Survival Rate. (Can they cooperate enough to keep everyone alive for 50 episodes?)

---

## Slide 3: The Architecture - The Environment Brain
**The environment is the protagonist.**
- **The Sentinel:** Monitors the 20-episode rolling survival rate. If agents master the environment (>85% survival), it fires a trigger.
- **The Injector:** Introduces "Scarcity Droughts", slashing the resource pool to 5% capacity to simulate real-world supply chain shocks.
- **World Memory:** The environment remembers agent strategies. If an agent solves a drought the same way twice, the next drought gets recursively harder.

---

## Slide 4: The Graph - "40 to 8"
*(Visual: Line graph showing Survival Rate over Episodes)*
- **Act 1 (Cooperation):** Agents easily hit 95% survival.
- **Act 2 (Collapse):** The Sentinel detects mastery. The Injector fires an L3 Scarcity Drought. Trust decays. Survival plummets to 50%.
- **Act 3 (Recovery):** 
    - *Untrained Heuristics:* Take 40 episodes to recover stability.
    - *Trained RL Agents:* Recognize the drought and recover in **8 episodes**.

---

## Slide 5: The Closing Line
"When you build an environment that fights back, you don't just test what an agent knows—you test how quickly it can adapt when everything goes wrong."

**HuecoEnv — Ready for OpenEnv.**
