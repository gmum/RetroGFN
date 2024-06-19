import gin
import torch
from torch import nn


@gin.configurable()
class OptimizerBase:
    """
    The base class for optimizers used in Trainer.

    Args:
        cls_name: the name of the optimizer class.
        kwargs: additional arguments to pass to the optimizer.
    """

    def __init__(self, cls_name: str, **kwargs):
        self.cls_name = cls_name
        self.optimizer: torch.optim.Optimizer = ...
        self.kwargs = kwargs

    def initialize(self, model: nn.Module):
        self.optimizer = getattr(torch.optim, self.cls_name)(model.parameters(), **self.kwargs)

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()
