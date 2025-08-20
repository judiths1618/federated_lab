"""Aggregation utilities and strategies."""

from .aggregator import Aggregator
from .FedAvg import Strategy as FedAvg
from .CommitteeFedAvg import Strategy as CommitteeFedAvg
from .trimmed_mean import Strategy as TrimmedMean
from .flame import Strategy as Flame

__all__ = [
    "Aggregator",
    "FedAvg",
    "CommitteeFedAvg",
    "TrimmedMean",
    "Flame",
]
