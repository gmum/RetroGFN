from typing import Any, Dict

import gin
import torch


@gin.configurable()
class LRScheduler:
    """
    A wrapper around torch.optim.lr_scheduler.LRScheduler used in Trainer.
    """

    def __init__(self, cls_name: str, **kwargs: Dict[str, Any]):
        self.cls_name = cls_name
        self.kwargs = kwargs
        self.lr_scheduler: torch.optim.lr_scheduler.LRScheduler = ...

    def initialize(self, optimizer: torch.optim.Optimizer):
        self.lr_scheduler = getattr(torch.optim.lr_scheduler, self.cls_name)(
            optimizer, **self.kwargs
        )

    def step(self):
        self.lr_scheduler.step()
