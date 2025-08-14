from dataclasses import dataclass
from typing import Dict, Iterable


@dataclass
class DetectionCfg:
    """Configuration for malicious detection.

    Attributes
    ----------
    threshold: float
        Score below which a node is considered malicious.
    """

    threshold: float = 0.0


class MaliciousDetector:
    """Detect malicious nodes based on contribution scores.

    This utility helps compare detected malicious nodes against
    ground truth labels for analysis.
    """

    def __init__(self, cfg: DetectionCfg | None = None):
        self.cfg = cfg or DetectionCfg()

    def detect(self, scores: Dict[int, float]) -> Dict[int, bool]:
        """Flag nodes whose score falls below the configured threshold."""
        return {nid: score < self.cfg.threshold for nid, score in scores.items()}

    def compare(self, scores: Dict[int, float], malicious_ids: Iterable[int]) -> Dict[str, float]:
        """Compare detection results with ground truth.

        Parameters
        ----------
        scores: mapping from node id to contribution score
        malicious_ids: ids of nodes known to be malicious

        Returns
        -------
        dict with counts of tp/fp/tn/fn plus precision and recall
        """
        detections = self.detect(scores)
        malicious_set = set(malicious_ids)
        tp = sum(1 for nid, flag in detections.items() if flag and nid in malicious_set)
        fp = sum(1 for nid, flag in detections.items() if flag and nid not in malicious_set)
        fn = sum(1 for nid in malicious_set if not detections.get(nid, False))
        tn = sum(1 for nid, flag in detections.items() if not flag and nid not in malicious_set)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": precision,
            "recall": recall,
        }
