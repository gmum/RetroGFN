import pytest
from gfns.helpers.proxy_test_helpers import (
    helper__test_proxy__is_deterministic,
    helper__test_proxy__returns_sensible_values,
)

from gflownet.gfns.retro.proxies.retro_rfm_proxy import RetroRFMProxy

from .fixtures import *


@pytest.mark.parametrize("n_trajectories", [1, 10])
def test__retro_dataset_proxy__returns_sensible_values(
    retro_rfm_proxy: RetroRFMProxy, retro_env: RetroEnv, n_trajectories: int
):
    helper__test_proxy__returns_sensible_values(
        retro_env, retro_rfm_proxy, n_trajectories=n_trajectories
    )


@pytest.mark.parametrize("n_trajectories", [1, 10])
def test__retro_dataset_proxy__is_deterministic(
    retro_rfm_proxy: RetroRFMProxy, retro_env: RetroEnv, n_trajectories: int
):
    helper__test_proxy__is_deterministic(retro_env, retro_rfm_proxy, n_trajectories=n_trajectories)
