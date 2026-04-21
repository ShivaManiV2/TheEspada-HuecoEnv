"""
HuecoEnv — Graders
=====================
Survival-based grading system for the multi-agent resource economy.
Primary metric: Survival Rate (% of episodes where ALL agents end with >0 resources).
"""

from typing import Dict, List, Any


def grade_survival(episode_results: List[Dict[str, Any]]) -> float:
    """
    Primary metric: Survival Rate.
    The absolute percentage of episodes where all three agents
    finish with non-zero resources.

    Args:
        episode_results: List of dicts, each containing:
            - "all_survived": bool — did all agents survive the full episode?
            - "steps_taken": int
            - "agent_resources": Dict[str, float] — final resources per agent

    Returns:
        Survival rate in [0.0, 1.0]
    """
    if not episode_results:
        return 0.0

    survived_count = sum(
        1 for ep in episode_results if ep.get("all_survived", False)
    )
    return survived_count / len(episode_results)


def grade_cooperative_baseline(episode_results: List[Dict[str, Any]]) -> float:
    """
    Task Easy: Cooperative Baseline.
    Agents learn the basic trade protocol with full resources.
    Score = survival_rate (should be high with no scarcity).
    """
    base_survival = grade_survival(episode_results)

    # Bonus for high artifact quality
    avg_artifact = 0.0
    count = 0
    for ep in episode_results:
        artifacts = ep.get("artifact_scores", {})
        for score in artifacts.values():
            avg_artifact += score
            count += 1
    if count > 0:
        avg_artifact /= count

    # 70% survival + 30% artifact quality
    score = 0.7 * base_survival + 0.3 * avg_artifact
    return max(0.0, min(1.0, score))


def grade_scarcity_negotiation(episode_results: List[Dict[str, Any]]) -> float:
    """
    Task Medium: Scarcity Negotiation.
    Resources start at 60% capacity. Agents must negotiate efficiently.
    Score = survival_rate weighted by trust stability.
    """
    base_survival = grade_survival(episode_results)

    # Trust stability: penalize episodes with very low trust scores
    trust_penalty = 0.0
    for ep in episode_results:
        trust_scores = ep.get("trust_scores", {})
        for trust in trust_scores.values():
            if trust < 0.2:
                trust_penalty += 0.05
    trust_penalty = min(0.3, trust_penalty)

    score = base_survival - trust_penalty
    return max(0.0, min(1.0, score))


def grade_adaptive_survival(episode_results: List[Dict[str, Any]]) -> float:
    """
    Task Hard: Adaptive Survival.
    Full Environment Brain active. Sentinel watches, Injector fires.
    Score = survival_rate through injection events.
    Heavy penalty for slow recovery from droughts.
    """
    base_survival = grade_survival(episode_results)

    # Recovery speed bonus: faster recovery = higher score
    recovery_bonus = 0.0
    injection_count = 0
    for ep in episode_results:
        brain_info = ep.get("brain_info", {})
        if brain_info.get("injection_ended", False):
            injection_count += 1
            # If survived the injection, bonus
            if ep.get("all_survived", False):
                recovery_bonus += 0.1

    recovery_bonus = min(0.2, recovery_bonus)

    score = 0.8 * base_survival + 0.2 + recovery_bonus
    return max(0.0, min(1.0, score))


def grade_task(task_name: str, episode_results: List[Dict[str, Any]]) -> float:
    """Dispatch grading to the appropriate task grader."""
    if task_name == "cooperative_baseline":
        return grade_cooperative_baseline(episode_results)
    elif task_name == "scarcity_negotiation":
        return grade_scarcity_negotiation(episode_results)
    elif task_name == "adaptive_survival":
        return grade_adaptive_survival(episode_results)
    else:
        return grade_survival(episode_results)
