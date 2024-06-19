from gflownet import RandomSampler, UniformPolicy
from gflownet.api.env_base import EnvBase, TAction, TState
from gflownet.shared.policies.uniform_policy import TIndexedActionSpace
from gflownet.utils.helpers import seed_everything


def helper__test_env__forward_backward_consistency(
    env: EnvBase[TState, TIndexedActionSpace, TAction], n_trajectories: int
):
    """
    A helper function that tests whether the forward pass of the environment can be obtained by applying the backward
    pass to the last states of the trajectories.
    Args:
        env: environment to be tested
        n_trajectories: number of trajectories to sample

    Returns:
        None
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
    non_source_states = trajectories.get_non_source_states_flat()
    non_last_states = trajectories.get_non_last_states_flat()
    actions = trajectories.get_actions_flat()
    forward_action_spaces = trajectories.get_forward_action_spaces_flat()
    backward_action_spaces = trajectories.get_backward_action_spaces_flat()
    new_states = env.apply_backward_actions(non_source_states, actions)

    assert non_last_states == new_states
    for action, forward_action_space in zip(actions, forward_action_spaces):
        assert forward_action_space.is_action_allowed(action)

    for action, backward_action_space in zip(actions, backward_action_spaces):
        assert backward_action_space.is_action_allowed(action)


def helper__test_env__backward_forward_consistency(
    env: EnvBase[TState, TIndexedActionSpace, TAction],
    n_trajectories: int,
    sample_from_env: bool = True,
):
    """
    A helper function that tests whether the backward pass of the environment can be obtained by applying the forward
    pass to the source states of the obtained trajectories.
    Args:
        env: environment to be tested
        n_trajectories: number of trajectories to sample
        sample_from_env: whether to sample trajectories directly from the reversed env (assumes that the env has can
        implements `sample_from_terminal_states` method) or sample from the original env and then sample from the
        reversed env using the last states of the obtained trajectories.

    Returns:
        None
    """
    seed_everything(42)

    if sample_from_env:
        sampler = RandomSampler(
            policy=UniformPolicy(),
            env=env.reversed(),
            reward=None,
        )

        trajectories = next(
            iter(
                sampler.get_trajectories_iterator(
                    n_total_trajectories=n_trajectories, batch_size=-1
                )
            )
        )
    else:
        sampler = RandomSampler(
            policy=UniformPolicy(),
            env=env,
            reward=None,
        )

        trajectories = next(
            iter(
                sampler.get_trajectories_iterator(
                    n_total_trajectories=n_trajectories, batch_size=-1
                )
            )
        )
        last_states = trajectories.get_last_states_flat()
        reverse_sampler = RandomSampler(
            policy=UniformPolicy(),
            env=env.reversed(),
            reward=None,
        )
        trajectories = reverse_sampler.sample_trajectories_from_sources(last_states)

    non_source_states = trajectories.get_non_source_states_flat()
    non_last_states = trajectories.get_non_last_states_flat()
    actions = trajectories.get_actions_flat()
    forward_action_spaces = trajectories.get_forward_action_spaces_flat()
    backward_action_spaces = trajectories.get_backward_action_spaces_flat()
    new_states = env.apply_forward_actions(non_last_states, actions)

    assert non_source_states == new_states
    for action, forward_action_space in zip(actions, forward_action_spaces):
        assert forward_action_space.is_action_allowed(action)

    for action, backward_action_space in zip(actions, backward_action_spaces):
        assert backward_action_space.is_action_allowed(action)
