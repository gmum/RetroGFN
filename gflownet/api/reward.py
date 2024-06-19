import math
from dataclasses import dataclass
from typing import Dict, Generic, List, Literal

import gin
import torch
from torchtyping import TensorType

from gflownet.api.env_base import TState
from gflownet.api.proxy_base import ProxyBase


@dataclass
class RewardOutput:
    """
    The class to store the output obtained by calculating the reward on a batch of states. It contains the log rewards,
        rewards, proxy values, and possibly some components of the proxy values that may be used to compute metrics.

    Attributes:
        log_reward: The log rewards.
        reward: The rewards.
        proxy: The proxy values.
        proxy_components: A dictionary of components of the proxy values. It is used to compute metrics. If None, the
            proxy values have no components.
    """

    log_reward: TensorType[float]
    reward: TensorType[float]
    proxy: TensorType[float]
    proxy_components: Dict[str, TensorType[float]] | None = None

    def set_device(self, device: str):
        """
        Set the device on which to perform the computations.

        Args:
            device: a string representing the device.

        Returns:
            None
        """
        self.log_reward = self.log_reward.to(device)
        self.reward = self.reward.to(device)
        self.proxy = self.proxy.to(device)
        if self.proxy_components is not None:
            for key, value in self.proxy_components.items():
                self.proxy_components[key] = value.to(device)

    @classmethod
    def from_list(cls, items: List["RewardOutput"]) -> "RewardOutput":
        """
        Concatenate a list of RewardOutput objects into a single RewardOutput object.
        Used in `Trajectories.from_trajectories` method.

        Args:`
            items: a list of RewardOutput objects.

        Returns:
            a new RewardOutput object that is the concatenation of the input items.
        """
        proxy_components = [
            item.proxy_components for item in items if item.proxy_components is not None
        ]
        if len(proxy_components) == 0:
            new_proxy_components = None
        elif len(proxy_components) == len(items):
            new_proxy_components = {}
            for key in proxy_components[0].keys():
                new_proxy_components[key] = torch.cat([item[key] for item in proxy_components])
        else:
            raise ValueError("Some items have proxy components and some don't!")
        return cls(
            log_reward=torch.cat([item.log_reward for item in items]),
            reward=torch.cat([item.reward for item in items]),
            proxy=torch.cat([item.proxy for item in items]),
            proxy_components=new_proxy_components,
        )

    def masked_select(self, mask: TensorType[bool]) -> "RewardOutput":
        """
        Select a subset of the RewardOutput object using a boolean mask. Used in `Trajectories.masked_select` method.

        Args:
            mask: a boolean mask of shape `(n,)` where `n` is the number of elements in the RewardOutput object.

        Returns:
            a new RewardOutput object that contains only the elements selected by the mask.
        """
        if self.proxy_components is not None:
            proxy_components = {
                key: value[mask].clone() for key, value in self.proxy_components.items()
            }
        else:
            proxy_components = None
        return RewardOutput(
            log_reward=self.log_reward[mask].clone(),
            reward=self.reward[mask].clone(),
            proxy=self.proxy[mask].clone(),
            proxy_components=proxy_components,
        )


@gin.configurable()
class Reward(Generic[TState]):
    """
    A class representing the reward function. The reward function is a function that takes a batch of states and
        computes rewards that are used to train the policy.

    Type parameters:
        TState: The type of the states.

    Attributes:
        proxy: The proxy that is used to compute the rewards.
        reward_boosting: The type of reward boosting. It can be either "linear" or "exponential".
        min_reward: The minimum reward value. If the reward boosting is "linear", the rewards are clamped to be at least
            `min_reward`. If the reward boosting is "exponential", the log rewards are clamped to be at least
            `math.log(min_reward)`.
        beta: The coefficient that multiplies the proxy values to compute the rewards.
    """

    def __init__(
        self,
        proxy: ProxyBase[TState],
        reward_boosting: Literal["linear", "exponential"] = "linear",
        min_reward: float = 0.0,
        beta: float = 1.0,
    ):
        assert reward_boosting in ["linear", "exponential"]
        if reward_boosting == "linear" and not proxy.is_non_negative:
            raise ValueError("Reward boosting is linear but proxy is not non-negative!")
        self.proxy = proxy
        self.reward_boosting = reward_boosting
        self.min_reward = min_reward
        self.min_log_reward = math.log(min_reward) if min_reward > 0 else -float("inf")
        self.beta = beta

    @torch.no_grad()
    def compute_reward_output(self, states: List[TState]) -> RewardOutput:
        """
        Compute the reward output on a batch of states.

        Args:
            states: a list of states of length `n_states`.

        Returns:
            a `RewardOutput` object. The reward output contains the log rewards, rewards, proxy values,
            and possibly some components of the proxy values that may be used to compute metrics.
        """
        proxy_output = self.proxy.compute_proxy_output(states)
        value = proxy_output.value
        signed_value = value if self.proxy.higher_is_better else -value
        if self.reward_boosting == "linear":
            reward = signed_value * self.beta
            reward = torch.clamp(reward, min=self.min_reward)
            log_reward = reward.log()
        else:
            log_reward = signed_value * self.beta
            log_reward = torch.clamp(log_reward, min=self.min_log_reward)
            reward = log_reward.exp()
        return RewardOutput(
            log_reward=log_reward,
            reward=reward,
            proxy=value,
            proxy_components=proxy_output.components,
        )

    def set_device(self, device: str):
        """
        Set the device on which to perform the computations.

        Args:
            device: a string representing the device.

        Returns:
            None
        """
        self.proxy.set_device(device)
