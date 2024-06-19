import pytest
from gfns.helpers.policy_test_helpers import (
    helper__test_policy__returns_sensible_log_probs,
    helper__test_policy__samples_only_allowed_actions,
)

from gflownet import ForwardHyperGridPolicy, HyperGridEnv


@pytest.mark.parametrize("size", [4, 16])
@pytest.mark.parametrize("n_dimensions", [2, 3])
@pytest.mark.parametrize("max_num_steps", [2, 3])
@pytest.mark.parametrize("n_trajectories", [1, 10])
def test__hypergrid_policy__returns_only_allowed_action(
    size: int, n_dimensions: int, max_num_steps: int, n_trajectories: int
):
    hypergrid_env = HyperGridEnv(size=size, n_dimensions=n_dimensions, max_num_steps=max_num_steps)
    forward_policy = ForwardHyperGridPolicy(env=hypergrid_env)
    helper__test_policy__samples_only_allowed_actions(forward_policy, hypergrid_env, n_trajectories)


@pytest.mark.parametrize("size", [4, 16])
@pytest.mark.parametrize("n_dimensions", [2, 3])
@pytest.mark.parametrize("max_num_steps", [2, 3])
@pytest.mark.parametrize("n_trajectories", [1, 10])
def test__hypergrid_policy__returns_sensible_log_probs(
    size: int, n_dimensions: int, max_num_steps: int, n_trajectories: int
):
    hypergrid_env = HyperGridEnv(size=size, n_dimensions=n_dimensions, max_num_steps=max_num_steps)
    forward_policy = ForwardHyperGridPolicy(env=hypergrid_env)
    helper__test_policy__returns_sensible_log_probs(forward_policy, hypergrid_env, n_trajectories)
