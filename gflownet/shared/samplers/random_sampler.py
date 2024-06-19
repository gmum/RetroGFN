from typing import Generic, Iterator

import gin

from gflownet.api.env_base import EnvBase, TAction, TActionSpace, TState
from gflownet.api.policy_base import PolicyBase
from gflownet.api.reward import Reward
from gflownet.api.sampler_base import SamplerBase
from gflownet.api.trajectories import Trajectories


@gin.configurable()
class RandomSampler(
    SamplerBase[TState, TActionSpace, TAction], Generic[TState, TActionSpace, TAction]
):
    """
    A standard sampler that samples trajectories from the environment using a policy.
    """

    def __init__(
        self,
        policy: PolicyBase[TState, TActionSpace, TAction],
        env: EnvBase[TState, TActionSpace, TAction],
        reward: Reward[TState] | None,
    ):
        super().__init__(policy, env, reward)

    def sample_trajectories(
        self, n_trajectories: int
    ) -> Trajectories[TState, TActionSpace, TAction]:
        """
        Sample n_trajectories from the environment using the policy.
        Args:
            n_trajectories: the number of trajectories to sample.
        Returns:
            the sampled trajectories.
        """
        source_states = self.env.sample_source_states(n_trajectories)
        return self.sample_trajectories_from_sources(source_states)

    def get_trajectories_iterator(
        self, n_total_trajectories: int, batch_size: int
    ) -> Iterator[Trajectories[TState, TActionSpace, TAction]]:
        """
        Get an iterator that samples trajectories from the environment. It can be used to sampled trajectories in
            batched manner.
        Args:
            n_total_trajectories: total number of trajectories to sample. If set to -1, the sampler should iterate over
                all source states (used in `SequentialSampler`).
            batch_size: the size of the batch. If -1, the batch size is equal to the number of n_total_trajectories.

        Returns:
            an iterator that samples trajectories.
        """
        batch_size = n_total_trajectories if batch_size == -1 else batch_size
        batches_sizes = [batch_size] * (n_total_trajectories // batch_size)
        if n_total_trajectories % batch_size:
            batches_sizes.append(n_total_trajectories % batch_size)
        for batch_size in batches_sizes:
            yield self.sample_trajectories(batch_size)
