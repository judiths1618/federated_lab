"""Aggregation utilities and strategies."""

from .aggregator import Aggregator
from .fedavg import Strategy as FedAvg
from .committee_fedavg import Strategy as CommitteeFedAvg
from .trimmed_mean import Strategy as TrimmedMean

__all__ = [
    "Aggregator",
    "FedAvg",
    "CommitteeFedAvg",
    "TrimmedMean",
]
