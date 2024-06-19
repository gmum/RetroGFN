from typing import Generic, Iterator

import gin
import numpy as np
from more_itertools import chunked

from gflownet.api.env_base import EnvBase, TAction, TActionSpace, TState
from gflownet.api.policy_base import PolicyBase
from gflownet.api.reward import Reward
from gflownet.api.sampler_base import SamplerBase
from gflownet.api.trajectories import Trajectories


@gin.configurable()
class SequentialSampler(
    SamplerBase[TState, TActionSpace, TAction], Generic[TState, TActionSpace, TAction]
):
    """
    A sampler that samples trajectories from the environment using a policy in a sequential manner. It assumes that
    the environment has a fixed number of source states (e.g. some pre-defined sets of molecules) and samples
    trajectories from each source `n_repeats` times. It can be used to evaluate the performance of some
    conditioned GFlowNets.

    Attributes:
        policy: A policy that will be used sample actions.
        env: An environment that describes the transitions between states.
        reward: A reward function that assigns rewards to the terminal states.
        n_repeats: the number of times to sample trajectories from each source state.
    """

    def __init__(
        self,
        policy: PolicyBase[TState, TActionSpace, TAction],
        env: EnvBase[TState, TActionSpace, TAction],
        reward: Reward[TState],
        n_repeats: int,
    ):
        super().__init__(policy, env, reward)
        self.n_repeats = n_repeats
        self.n_source_states = self.env.get_num_source_states()

    def get_trajectories_iterator(
        self, n_total_trajectories: int, batch_size: int
    ) -> Iterator[Trajectories]:
        """
        Get an iterator that samples trajectories from the environment. It can be used to sampled trajectories in
            batched manner.
        Args:
            n_total_trajectories: it must be set to -1, indicating that the sampler should iterate over all source states.
            batch_size: the size of the batch. If -1, the batch size is equal to the number of
                self.n_source_states. Note that the actual batch size used during the sampling is equal to
                `batch_size * n_repeats`.

        Returns:
            an iterator that samples trajectories. The number of total sampled trajectories is actually equal to
                n_total_trajectories * n_repeats`.
        """
        assert n_total_trajectories == -1, "SequentialSampler should iterate over all source states"
        batch_size = self.n_source_states if batch_size == -1 else batch_size
        indices = np.arange(self.n_source_states)
        for indices_batch in chunked(indices, batch_size):
            indices_batch = np.repeat(indices_batch, self.n_repeats)
            source_states = self.env.get_source_states_at_index(list(indices_batch))
            yield self.sample_trajectories_from_sources(source_states)
