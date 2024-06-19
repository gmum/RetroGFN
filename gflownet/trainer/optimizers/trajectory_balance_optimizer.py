import gin
import torch
from torch import nn

from .optimizer_base import OptimizerBase


@gin.configurable()
class TrajectoryBalanceOptimizer(OptimizerBase):
    """
    An optimizer that balances the learning rates of the logZ parameters and the other parameters.

    Args:
        cls_name: the name of the optimizer class.
        lr: the learning rate.
        logZ_multiplier: the multiplier for the learning rate of the logZ parameters.
        kwargs: additional arguments to pass to the optimizer.
    """

    def __init__(self, cls_name: str, lr: float, logZ_multiplier: float = 1.0, **kwargs):
        super().__init__(cls_name)
        self.lr = lr
        self.logZ_multiplier = logZ_multiplier
        self.kwargs = kwargs

    def initialize(self, model: nn.Module):
        parameters = [
            {
                "params": [p for n, p in model.named_parameters() if "logZ" in n],
                "lr": self.lr * self.logZ_multiplier,
            },
            {
                "params": [p for n, p in model.named_parameters() if "logZ" not in n],
                "lr": self.lr,
            },
        ]
        self.optimizer = getattr(torch.optim, self.cls_name)(parameters, **self.kwargs)
