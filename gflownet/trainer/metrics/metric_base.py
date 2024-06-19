from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence

from gflownet.api.trajectories import Trajectories


class MetricsBase(ABC):
    """
    The base class for metrics used in Trainer.
    """

    @abstractmethod
    def compute_metrics(self, trajectories: Trajectories) -> Dict[str, Any]:
        ...


class MetricsList(MetricsBase):
    def __init__(self, metrics: Sequence[MetricsBase]):
        self.metrics = metrics

    def compute_metrics(self, trajectories: Trajectories) -> Dict[str, Any]:
        metrics = {}
        for metric in self.metrics:
            metrics.update(metric.compute_metrics(trajectories))
        return metrics
