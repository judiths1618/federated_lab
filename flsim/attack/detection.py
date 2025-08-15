from dataclasses import dataclass
from typing import Dict
import torch


@dataclass
class AttackDetectionParams:
    """Thresholds for detecting suspicious updates.

    Attributes
    ----------
    update_threshold: float
        L2 distance threshold between a client's update and the aggregated
        update. Exceeding this marks the client as suspicious.
    acc_threshold: float
        Acceptable difference between claimed train accuracy and the
        aggregated model's accuracy.
    test_acc_threshold: float
        Acceptable difference between claimed test accuracy and the
        aggregated model's test accuracy.
    loss_threshold: float
        Acceptable difference between claimed training loss and the
        aggregated model's training loss.
    """

    update_threshold: float = 5.0
    acc_threshold: float = 0.2
    test_acc_threshold: float = 0.2
    loss_threshold: float = 1.0


class AttackDetector:
    """Detect malicious clients based on updates and claimed metrics."""

    def __init__(self, params: AttackDetectionParams | None = None):
        self.params = params or AttackDetectionParams()

    def detect(
        self,
        client_updates: Dict[int, Dict[str, torch.Tensor]],
        claimed_metrics: Dict[int, Dict[str, float]],
        base_model: Dict[str, torch.Tensor],
        aggregated_model: Dict[str, torch.Tensor],
        aggregated_metrics: Dict[str, float],
    ) -> Dict[int, bool]:
        """Return mapping from client id to detected status."""
        agg_delta = {k: aggregated_model[k] - base_model[k] for k in base_model}
        detections: Dict[int, bool] = {}
        for cid, update in client_updates.items():
            # Compute distance between client's update and aggregated update
            dist_sq = 0.0
            for k, agg_upd in agg_delta.items():
                if k in update:
                    diff = update[k] - agg_upd
                    dist_sq += float(torch.sum(diff * diff))
            dist = dist_sq ** 0.5
            metrics = claimed_metrics.get(cid, {})
            acc_diff = abs(metrics.get("acc", 0.0) - aggregated_metrics.get("acc", 0.0))
            test_acc_diff = abs(metrics.get("test_acc", 0.0) - aggregated_metrics.get("test_acc", 0.0))
            loss_diff = abs(metrics.get("train_loss", 0.0) - aggregated_metrics.get("train_loss", 0.0))
            is_mal = (
                dist > self.params.update_threshold
                or acc_diff > self.params.acc_threshold
                or test_acc_diff > self.params.test_acc_threshold
                or loss_diff > self.params.loss_threshold
            )
            detections[cid] = is_mal
        return detections
