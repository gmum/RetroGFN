from typing import Generic, Iterator

import gin
import torch
from torch import nn
from torch.nn import Parameter

from gflownet.api.env_base import TAction, TActionSpace, TState
from gflownet.api.objective_base import ObjectiveBase, ObjectiveOutput
from gflownet.api.policy_base import PolicyBase
from gflownet.api.trajectories import Trajectories


@gin.configurable()
class TrajectoryBalanceObjective(ObjectiveBase[TState, TActionSpace, TAction]):
    """
    Trajectory balance objective for GFN from the paper "Trajectory balance: Improved credit assignment in GFlowNets"
    (https://arxiv.org/abs/2201.13259)

    Attributes:
        forward_policy: a policy that estimates the probabilities of actions taken in the forward direction.
        backward_policy: a policy that estimates the probabilities of actions taken in the backward direction.
        logZ: a learnable parameter that represents the log partition function.
    """

    def __init__(
        self,
        forward_policy: PolicyBase[TState, TActionSpace, TAction],
        backward_policy: PolicyBase[TState, TActionSpace, TAction],
    ):
        super().__init__(forward_policy=forward_policy, backward_policy=backward_policy)
        self.logZ = nn.Parameter(torch.zeros(1).float())
        self.device = "cpu"

    def compute_objective_output(
        self, trajectories: Trajectories[TState, TActionSpace, TAction]
    ) -> ObjectiveOutput:
        """
        Compute the objective output on a batch of trajectories.

        Args:
            trajectories: the batch of trajectories obtained in the sampling process. It contains the states, actions,
                action spaces in forward and backward directions, and rewards. Other important quantities (e.g. log
                probabilities of taking actions in forward and backward directions) should be assigned in this method
                using appropriate methods (e.g. assign_log_probs).

        Returns:
            The output of the objective function, containing the loss and possibly some metrics.
        """
        self.assign_log_probs(trajectories)

        forward_log_prob = trajectories.get_forward_log_probs_flat()  # [n_actions]
        backward_log_prob = trajectories.get_backward_log_probs_flat()  # [n_actions]
        log_reward = trajectories.get_reward_outputs().log_reward  # [n_trajectories]
        index = trajectories.get_index_flat().to(self.device)  # [n_actions]

        loss = torch.scatter_add(
            input=self.logZ.sum() - log_reward,
            index=index,
            src=forward_log_prob - backward_log_prob,
            dim=0,
        )  # [n_trajectories]
        loss = loss.pow(2).mean()
        return ObjectiveOutput(loss=loss, metrics={"logZ": self.logZ.item()})

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """
        Get the parameters of the objective function.

        Args:
            recurse: whether to recursively get the parameters of the submodules.

        Returns:
            An iterator over the parameters.
        """
        yield from super().parameters(recurse)
        yield self.logZ

    def set_device(self, device: str):
        """
        Set the device on which to perform the computations.
        Args:
            device: device to set.

        Returns:
            None
        """
        super().set_device(device)
        self.logZ.to(device)
