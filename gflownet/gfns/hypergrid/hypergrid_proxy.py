from typing import List

import gin
import torch

from gflownet.api.proxy_base import ProxyBase, ProxyOutput

from .hypergrid_env import HyperGridState


@gin.configurable()
class HyperGridProxy(ProxyBase[HyperGridState]):
    """
    A proxy that computes the reward for the hypergrid environment. It originates in the paper "Flow Network based
    Generative Models for Non-Iterative Diverse Candidate Generation" (https://arxiv.org/pdf/2106.04399)
    """

    def __init__(self, size: int, r_0: float = 0.01, r_1: float = 0.5, r_2: float = 2.0):
        self.size = size
        self.r_0 = r_0
        self.r_1 = r_1
        self.r_2 = r_2
        self.device = "cpu"

    def compute_proxy_output(self, states: List[HyperGridState]) -> ProxyOutput:
        coords = torch.tensor([state.coords for state in states]).float().to(self.device)
        coords = coords / (self.size - 1)
        ax = torch.abs(coords - 0.5)
        reward = (
            self.r_0
            + (0.25 < ax).prod(-1) * self.r_1
            + ((0.3 < ax) * (ax < 0.4)).prod(-1) * self.r_2
        )
        return ProxyOutput(value=reward)

    @property
    def is_non_negative(self) -> bool:
        return True

    @property
    def higher_is_better(self) -> bool:
        return True

    def set_device(self, device: str):
        self.device = device
