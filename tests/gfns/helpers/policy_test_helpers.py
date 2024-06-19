import torch

from gflownet import RandomSampler, UniformPolicy
from gflownet.api.env_base import EnvBase, TAction, TState
from gflownet.api.policy_base import PolicyBase
from gflownet.shared.policies.uniform_policy import TIndexedActionSpace
from gflownet.utils.helpers import seed_everything


def helper__test_policy__samples_only_allowed_actions(
    policy: PolicyBase[TState, TIndexedActionSpace, TAction],
    env: EnvBase[TState, TIndexedActionSpace, TAction],
    n_trajectories: int,
):
    """
    A helper function that tests whether the policy samples only allowed actions.

    Args:
        policy: a policy to be tested
        env: an environment corresponding to the policy
        n_trajectories: number of trajectories to sample
    """
    seed_everything(42)
    sampler = RandomSampler(
        policy=policy,
        env=env,
        reward=None,
    )

    trajectories = next(
        iter(sampler.get_trajectories_iterator(n_total_trajectories=n_trajectories, batch_size=-1))
    )
    actions = trajectories.get_actions_flat()
    forward_action_spaces = trajectories.get_forward_action_spaces_flat()
    backward_action_spaces = trajectories.get_backward_action_spaces_flat()

    for action, forward_action_space in zip(actions, forward_action_spaces):
        assert forward_action_space.is_action_allowed(action)

    for action, backward_action_space in zip(actions, backward_action_spaces):
        assert backward_action_space.is_action_allowed(action)


def helper__test_policy__returns_sensible_log_probs(
    policy: PolicyBase[TState, TIndexedActionSpace, TAction],
    env: EnvBase[TState, TIndexedActionSpace, TAction],
    n_trajectories: int,
):
    """
    A helper function that tests whether the policy returns sensible log probabilities for the sampled trajectories.

    Args:
        policy: a policy to be tested
        env: an environment corresponding to the policy
        n_trajectories: number of trajectories to sample
    """
    seed_everything(42)
    sampler = RandomSampler(
        policy=UniformPolicy(),
        env=env,
        reward=None,
    )

    trajectories = next(
        iter(sampler.get_trajectories_iterator(n_total_trajectories=n_trajectories, batch_size=-1))
    )

    states = trajectories.get_non_last_states_flat()
    actions = trajectories.get_actions_flat()
    forward_action_spaces = trajectories.get_forward_action_spaces_flat()
    log_probs = policy.compute_action_log_probs(states, forward_action_spaces, actions)

    assert torch.isnan(log_probs).sum() == 0
    assert torch.isinf(log_probs).sum() == 0
