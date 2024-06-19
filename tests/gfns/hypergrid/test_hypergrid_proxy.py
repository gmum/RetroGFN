import pytest
from gfns.helpers.proxy_test_helpers import helper__test_proxy__returns_sensible_values

from gflownet import HyperGridEnv, HyperGridProxy


@pytest.mark.parametrize("size", [4, 16])
@pytest.mark.parametrize("n_dimensions", [2, 3])
@pytest.mark.parametrize("max_num_steps", [2, 3])
@pytest.mark.parametrize("n_trajectories", [1, 10])
def test__hypergrid_proxy_returns_sensible_values(
    size: int, n_dimensions: int, max_num_steps: int, n_trajectories: int
):
    env = HyperGridEnv(size=size, n_dimensions=n_dimensions, max_num_steps=max_num_steps)
    proxy = HyperGridProxy(size=size)
    helper__test_proxy__returns_sensible_values(env, proxy, n_trajectories=n_trajectories)
