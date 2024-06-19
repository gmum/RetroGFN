from typing import Generic, Hashable, Iterator, List, Set, TypeVar

import gin
import numpy as np
import torch
from more_itertools import chunked

from gflownet.api.env_base import TAction, TActionSpace, TState
from gflownet.api.replay_buffer_base import ReplayBufferBase
from gflownet.api.sampler_base import SamplerBase
from gflownet.api.trajectories import Trajectories

THashableState = TypeVar("THashableState", bound=Hashable)


@gin.configurable()
class RewardPrioritizedReplayBuffer(ReplayBufferBase[THashableState, TActionSpace, TAction]):
    """
    A replay buffer that stores terminal states and their proxy values. The proxy values are used to weight the
    probability of sampling a backward trajectory starting from a terminal state. The proxy values are multiplied by a
    temperature coefficient before applying the softmax function to get the probabilities. It is inspired by the
    "An Empirical Study of the Effectiveness of Using a Replay Buffer on Mode Discovery in GFlowNets" paper.

    Args:
        sampler: a sampler that samples trajectories from the environment. The environment within the sampler should
            be reversed.
        max_size: the maximum number of terminal states to store.
        temperature: the temperature coefficient that is multiplied by the proxy values before applying the softmax
            function.
    """

    def __init__(
        self,
        sampler: SamplerBase[THashableState, TActionSpace, TAction],
        max_size: int = int(1e6),
        temperature: float = 1.0,
    ):
        assert sampler.env.is_reversed, "The environment should be reversed."
        self.sampler = sampler
        self.max_size = int(max_size)
        self.states_list: List[THashableState] = []
        self.states_set: Set[THashableState] = set()
        self.proxy_value_array = np.zeros((self.max_size + 1,), dtype=np.float32)
        self.temperature = temperature

    def get_trajectories_iterator(
        self, n_total_trajectories: int, batch_size: int
    ) -> Iterator[Trajectories[THashableState, TActionSpace, TAction]]:
        """
        Get an iterator that samples trajectories from the environment. It can be used to sampled trajectories in
            batched manner.
        Args:
            n_total_trajectories: total number of trajectories to sample.
            batch_size: the size of the batch. If -1, the batch size is equal to the number of n_total_trajectories.

        Returns:
            an iterator that samples trajectories.
        """
        n_total_trajectories = min(n_total_trajectories, self.size)
        batch_size = n_total_trajectories if batch_size == -1 else batch_size

        if n_total_trajectories == 0:
            yield Trajectories()
            return

        logits = torch.tensor(self.proxy_value_array[: self.size] * self.temperature)
        probs = torch.nn.functional.softmax(logits, dim=0).numpy()

        sampled_indices = np.random.choice(
            self.size, size=n_total_trajectories, replace=False, p=probs
        )
        sampled_states = [self.states_list[i] for i in sampled_indices]

        for sampled_states_chunk in chunked(sampled_states, batch_size):
            yield self.sampler.sample_trajectories_from_sources(sampled_states_chunk)

    def add_trajectories(self, trajectories: Trajectories[THashableState, TActionSpace, TAction]):
        """
        Add the terminal states from the trajectories to the replay buffer that are not already in the replay buffer.

        Args:
            trajectories: trajectories to get the terminal states from.

        Returns:
            None
        """
        terminal_states = trajectories.get_last_states_flat()
        proxy_value = trajectories.get_reward_outputs().proxy
        for state, proxy_value in zip(terminal_states, proxy_value):
            if state not in self.states_set:
                self._add_state(state, proxy_value.item())

    def state_dict(self) -> dict:
        return {
            "states_list": self.states_list,
            "proxy_value_array": self.proxy_value_array,
        }

    def load_state_dict(self, state_dict: dict):
        self.states_list = state_dict["states_list"]
        self.proxy_value_array = state_dict["proxy_value_array"]
        self.states_set = set(self.states_list)

    def _add_state(self, state: THashableState, proxy_value: float):
        self.proxy_value_array[self.size] = proxy_value
        self.states_list.append(state)
        self.states_set.add(state)
        if self.size > self.max_size:
            el = self.states_list.pop(0)
            self.states_set.remove(el)
            self.proxy_value_array[:-1] = self.proxy_value_array[1:]

    @property
    def size(self):
        return len(self.states_list)

    def set_device(self, device: str):
        self.sampler.set_device(device)

    def clear_action_embedding_cache(self):
        self.sampler.clear_action_embedding_cache()

    def clear_sampling_cache(self):
        self.sampler.clear_sampling_cache()
