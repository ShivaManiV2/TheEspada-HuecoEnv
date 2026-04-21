"""
HuecoEnv — Task Easy: Cooperative Baseline
=============================================
No injections, full resources.
Target: agents learn basic trade protocol and achieve high survival.
"""

from env.huecoenv_env import HuecoEnv
from tasks.graders import grade_cooperative_baseline


class CooperativeBaselineTask:
    def __init__(self):
        self.env = HuecoEnv()
        self.episode_results = []

    def get_env(self):
        return self.env

    def reset(self):
        return self.env.reset(task_name="cooperative_baseline")

    def step(self, actions: dict):
        return self.env.step(actions)

    def end_episode(self):
        """Call after each episode to record results."""
        state = self.env.state()
        result = {
            "all_survived": all(
                a.has_resources() for a in state.agents.values()
            ),
            "steps_taken": state.step,
            "agent_resources": {
                aid: a.compute_held + a.data_held
                for aid, a in state.agents.items()
            },
            "artifact_scores": {
                aid: a.artifact_score
                for aid, a in state.agents.items()
            },
            "trust_scores": {
                aid: a.trust_score
                for aid, a in state.agents.items()
            },
        }
        self.episode_results.append(result)
        brain_info = self.env.on_episode_end()
        result["brain_info"] = brain_info
        return result

    def grade(self) -> float:
        return grade_cooperative_baseline(self.episode_results)
