import pytest
from gfns.helpers.env_test_helpers import (
    helper__test_env__backward_forward_consistency,
    helper__test_env__forward_backward_consistency,
)

from gflownet import HyperGridEnv


@pytest.mark.parametrize("size", [4, 16])
@pytest.mark.parametrize("n_dimensions", [2, 3])
@pytest.mark.parametrize("max_num_steps", [2, 3])
@pytest.mark.parametrize("n_trajectories", [1, 10])
def test__retro_env__forward_backward_consistency(
    size: int, n_dimensions: int, max_num_steps: int, n_trajectories: int
):
    hypergrid_env = HyperGridEnv(size=size, n_dimensions=n_dimensions, max_num_steps=max_num_steps)
    helper__test_env__forward_backward_consistency(hypergrid_env, n_trajectories=n_trajectories)


@pytest.mark.parametrize("size", [4, 16])
@pytest.mark.parametrize("n_dimensions", [2, 3])
@pytest.mark.parametrize("max_num_steps", [2, 3])
@pytest.mark.parametrize("n_trajectories", [1, 10])
def test__retro_env__backward_forward_consistency(
    size: int, n_dimensions: int, max_num_steps: int, n_trajectories: int
):
    hypergrid_env = HyperGridEnv(size=size, n_dimensions=n_dimensions, max_num_steps=max_num_steps)
    helper__test_env__backward_forward_consistency(
        hypergrid_env, n_trajectories=n_trajectories, sample_from_env=False
    )
