from collections import defaultdict
import csv
from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass
class DetectionCfg:
    """Configuration for malicious detection.

    Attributes
    ----------
    threshold: float
        Score below which a node is considered malicious.
    penalty: float
        Amount by which a detected node is penalized.
    """

    threshold: float = 0.0
    penalty: float = 1.0


class MaliciousDetector:
    """Detect malicious nodes based on historical contribution scores.

    The detector maintains a history of scores per node. When the average
    score of a node falls below :class:`DetectionCfg.threshold`, the node is
    marked as malicious, a penalty is recorded and the node is considered
    disabled for subsequent rounds.
    """

    def __init__(self, cfg: DetectionCfg | None = None):
        self.cfg = cfg or DetectionCfg()
        self.history: Dict[int, List[float]] = defaultdict(list)
        self.penalties: Dict[int, float] = defaultdict(float)
        self.disabled_nodes: set[int] = set()

    def update_history(self, scores: Dict[int, float]) -> None:
        """Append current round scores to the per-node history."""
        for nid, score in scores.items():
            self.history[int(nid)].append(float(score))

    def update_history_from_log(self, log_path: str) -> None:
        """Populate history by reading a ``fl_log.csv`` file.

        The log is expected to contain ``node_id`` and ``contrib_score``
        columns. Missing or malformed rows are ignored.
        """
        with open(log_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    nid = int(row.get("node_id"))
                    score = float(row.get("contrib_score", 0.0))
                except (TypeError, ValueError):
                    continue
                self.history[nid].append(score)

    def detect(self, scores: Dict[int, float]) -> Dict[int, bool]:
        """Detect malicious nodes using historical behaviour.

        Parameters
        ----------
        scores: mapping from node id to contribution score for the current
            round. These values are appended to the internal history before
            detection.
        """
        self.update_history(scores)
        detections: Dict[int, bool] = {}
        for nid, hist in self.history.items():
            avg = sum(hist) / len(hist) if hist else 0.0
            is_mal = avg < self.cfg.threshold
            if is_mal and nid not in self.disabled_nodes:
                self.penalties[nid] += self.cfg.penalty
                self.disabled_nodes.add(nid)
            detections[nid] = is_mal
        return detections

    def is_disabled(self, node_id: int) -> bool:
        """Return ``True`` if the node was previously flagged malicious."""
        return int(node_id) in self.disabled_nodes

    def get_penalty(self, node_id: int) -> float:
        """Return the cumulative penalty applied to ``node_id``."""
        return float(self.penalties.get(int(node_id), 0.0))

    def compare(self, scores: Dict[int, float], malicious_ids: Iterable[int]) -> Dict[str, float]:
        """Compare detection results with ground truth.

        Parameters
        ----------
        scores: mapping from node id to contribution score for the current
            round. These values are appended to history prior to detection.
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
