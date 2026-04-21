"""
HuecoEnv — FastAPI Server
=============================
Multi-agent environment API with endpoints for reset, step, state,
metrics, and world memory. Serves the HuecoEnv dashboard.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
from pathlib import Path
import os

from env.huecoenv_env import HuecoEnv, ALL_AGENT_IDS
from agents.producer_agent import ProducerAgent
from agents.allocator_agent import AllocatorAgent
from agents.critic_agent import CriticAgent

app = FastAPI(title="HuecoEnv Environment API", version="2.0.0")

# Global environment instance
env_instance = HuecoEnv()
heuristic_agents = {
    "producer_0": ProducerAgent("producer_0"),
    "allocator_0": AllocatorAgent("allocator_0"),
    "critic_0": CriticAgent("critic_0"),
}


class StepRequest(BaseModel):
    actions: Dict[str, Dict[str, Any]]


class ResetRequest(BaseModel):
    task: Optional[str] = "cooperative_baseline"
    seed: Optional[int] = None


@app.get("/")
def home():
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"), status_code=200)
    return {"status": "HuecoEnv Environment API is running!"}


@app.post("/reset")
def reset(req: ResetRequest = None):
    task_name = req.task if req else "cooperative_baseline"
    seed = req.seed if req else None
    obs = env_instance.reset(task_name=task_name, seed=seed)
    return obs.model_dump()


@app.get("/reset")
def reset_get(task: str = "cooperative_baseline"):
    obs = env_instance.reset(task_name=task)
    return obs.model_dump()


@app.post("/step")
def step(req: StepRequest):
    try:
        obs, rewards, done, info = env_instance.step(req.actions)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    response = {
        "observation": obs.model_dump(),
        "rewards": rewards,
        "done": done,
        "info": {
            "step": info.get("step", 0),
            "all_alive": info.get("all_alive", False),
            "artifact_scores": info.get("artifact_scores", {}),
            "brain_state": info.get("brain_state", {}),
        },
    }
    return response


@app.post("/step-heuristic")
def step_heuristic():
    """Step the environment using internal heuristic agents."""
    obs = env_instance._get_observation()
    actions = {}
    for aid, agent in heuristic_agents.items():
        agent_obs = obs.observations.get(aid)
        if agent_obs:
            actions[aid] = agent.act(agent_obs).model_dump()
        else:
            actions[aid] = {"compute": 5.0, "data": 3.0, "want": {"score_share": 0.2}}

    try:
        obs, rewards, done, info = env_instance.step(actions)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "observation": obs.model_dump(),
        "rewards": rewards,
        "done": done,
        "info": {
            "step": info.get("step", 0),
            "all_alive": info.get("all_alive", False),
            "artifact_scores": info.get("artifact_scores", {}),
            "brain_state": info.get("brain_state", {}),
        },
    }


@app.get("/state")
def state():
    state_obj = env_instance.state()
    return state_obj.model_dump()


@app.get("/metrics")
def metrics():
    """Return survival curve data and key metrics."""
    state_obj = env_instance.state()
    brain_state = env_instance.brain.get_state()

    survival_history = state_obj.survival_history
    # Compute rolling survival rate
    window = 20
    rolling_rates = []
    for i in range(len(survival_history)):
        start = max(0, i - window + 1)
        chunk = survival_history[start:i + 1]
        rate = sum(1 for s in chunk if s) / len(chunk)
        rolling_rates.append(round(rate, 4))

    return {
        "survival_rate": state_obj.survival_rate,
        "survival_history": survival_history,
        "rolling_survival_rates": rolling_rates,
        "total_episodes": brain_state.get("current_episode", 0),
        "total_injections": brain_state.get("total_injections", 0),
        "injection_active": brain_state.get("injection_active", False),
        "injection_level": brain_state.get("injection_level", 0),
    }


@app.get("/world-memory")
def world_memory():
    """Return the Environment Brain's World Memory log."""
    return env_instance.brain.world_memory.model_dump()


@app.post("/episode-end")
def episode_end():
    """Notify the environment that an episode has ended."""
    brain_info = env_instance.on_episode_end()
    return brain_info


def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    print(f"Dashboard available at: http://127.0.0.1:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
