from gfns.helpers.proxy_test_helpers import helper__test_proxy__returns_sensible_values

from .fixtures import *


@pytest.mark.parametrize("n_trajectories", [1, 10])
def test__retro_dataset_proxy__returns_sensible_values(
    retro_dataset_proxy: RetroDatasetProxy, retro_env: RetroEnv, n_trajectories: int
):
    helper__test_proxy__returns_sensible_values(
        retro_env, retro_dataset_proxy, n_trajectories=n_trajectories
    )
