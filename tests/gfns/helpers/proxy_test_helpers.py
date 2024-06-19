import torch

from gflownet import RandomSampler, UniformPolicy
from gflownet.api.env_base import EnvBase, TAction, TState
from gflownet.api.proxy_base import ProxyBase, ProxyOutput
from gflownet.shared.policies.uniform_policy import TIndexedActionSpace
from gflownet.shared.proxies.cached_proxy import CachedProxyBase
from gflownet.utils.helpers import seed_everything


def helper__test_proxy__returns_sensible_values(
    env: EnvBase[TState, TIndexedActionSpace, TAction],
    proxy: ProxyBase[TState],
    n_trajectories: int,
):
    """
    A helper function that tests whether the proxy returns sensible values for the sampled trajectories.

    Args:
        env: an environment corresponding to the proxy
        proxy: a proxy to be tested
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

    states = trajectories.get_last_states_flat()
    proxy_output = proxy.compute_proxy_output(states)

    assert torch.isnan(proxy_output.value).sum() == 0
    assert torch.isinf(proxy_output.value).sum() == 0
    if proxy_output.components is not None:
        for component in proxy_output.components.values():
            assert torch.isnan(component).sum() == 0
            assert torch.isinf(component).sum() == 0

    if proxy.is_non_negative:
        assert (proxy_output.value < 0).sum() == 0


def helper__test_proxy__is_deterministic(
    env: EnvBase[TState, TIndexedActionSpace, TAction],
    proxy: ProxyBase[TState],
    n_trajectories: int,
):
    """
    A helper function that tests whether the proxy is deterministic.

    Args:
        env: an environment corresponding to the proxy
        proxy: a proxy to be tested
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

    states = trajectories.get_last_states_flat()
    proxy_output_1 = proxy.compute_proxy_output(states)
    if isinstance(proxy, CachedProxyBase):
        proxy.clear_cache()
    proxy_output_2 = proxy.compute_proxy_output(states)

    assert torch.allclose(proxy_output_1.value, proxy_output_2.value)
    if proxy_output_1.components is not None:
        for key in proxy_output_1.components.keys():
            assert torch.allclose(proxy_output_1.components[key], proxy_output_2.components[key])
