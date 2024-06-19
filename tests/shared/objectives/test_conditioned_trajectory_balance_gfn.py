import pytest
import torch
from shared.objectives.test_subtrajectory_balance_gfn import MockPolicy, MockProxy

from gflownet.api.reward import Reward
from gflownet.api.trajectories import Trajectories
from gflownet.shared.objectives import TrajectoryBalanceObjective
from gflownet.shared.objectives.conditioned_trajectory_balance_objective import (
    ConditionedTrajectoryBalanceObjective,
)


@pytest.fixture()
def reward() -> Reward:
    return Reward(proxy=MockProxy(), beta=1.0, reward_boosting="exponential", min_reward=0.0)


@pytest.fixture()
def objective() -> ConditionedTrajectoryBalanceObjective:
    policy = MockPolicy()
    return ConditionedTrajectoryBalanceObjective(
        forward_policy=policy,
        backward_policy=policy,
    )


def test__trajectory_balance_gfn__single_trajectory(
    reward: Reward, objective: ConditionedTrajectoryBalanceObjective
):
    trajectories = Trajectories()
    trajectories._states_list = [[3, 2, 4, 6]]
    trajectories._forward_action_spaces_list = [[[0, 1, 2], [0, 1, 2], [0, 1, 2]]]
    trajectories._backward_action_spaces_list = [[[0, 1, 2], [0, 1, 2], [0, 1, 2]]]
    trajectories._actions_list = [[2, 2, 2]]
    trajectories._reward_outputs = reward.compute_reward_output(trajectories.get_last_states_flat())

    loss = objective.compute_objective_output(trajectories).loss
    expected_loss = torch.tensor(4.0)

    assert torch.isclose(loss, expected_loss)


def test__trajectory_balance_gfn__many_trajectories(
    reward: Reward, objective: TrajectoryBalanceObjective
):
    trajectories = Trajectories()
    trajectories._states_list = [[2, 2, 4, 6], [0, 3, 6]]
    trajectories._forward_action_spaces_list = [
        [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
        [[0, 1, 2], [0, 1, 2]],
    ]
    trajectories._backward_action_spaces_list = [
        [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
        [[0, 1, 2], [0, 1, 2]],
    ]
    trajectories._actions_list = [[2, 2, 2], [3, 3]]
    trajectories._reward_outputs = reward.compute_reward_output(trajectories.get_last_states_flat())

    loss = objective.compute_objective_output(trajectories).loss
    expected_loss = torch.tensor(1.0)

    assert torch.isclose(loss, expected_loss)
