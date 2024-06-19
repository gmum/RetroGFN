# can return sensible log probs for any trajectory

from gfns.helpers.policy_test_helpers import (
    helper__test_policy__returns_sensible_log_probs,
    helper__test_policy__samples_only_allowed_actions,
)

from .fixtures import *


@pytest.mark.parametrize("n_trajectories", [1, 10])
def test__retro_policy__samples_only_allowed_actions(
    retro_forward_policy: RetroForwardPolicy, retro_env: RetroEnv, n_trajectories: int
):
    helper__test_policy__samples_only_allowed_actions(
        retro_forward_policy, retro_env, n_trajectories
    )


@pytest.mark.parametrize("n_trajectories", [1, 10])
def test__retro_policy__returns_sensible_log_probs(
    retro_forward_policy: RetroForwardPolicy, retro_env: RetroEnv, n_trajectories: int
):
    helper__test_policy__returns_sensible_log_probs(retro_forward_policy, retro_env, n_trajectories)
