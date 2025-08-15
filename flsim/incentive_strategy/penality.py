"""Penalty application utilities."""

def apply_penalty(stake: float, penalty: float) -> float:
    """Apply a penalty to a stake value, ensuring non-negative result."""
    return max(0.0, stake - penalty)

__all__ = ["apply_penalty"]
