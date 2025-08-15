"""Reward calculation utilities."""

def calculate_reward(contribution: float, base_reward: float = 1.0) -> float:
    """Simple proportional reward calculator."""
    return contribution * base_reward

__all__ = ["calculate_reward"]
