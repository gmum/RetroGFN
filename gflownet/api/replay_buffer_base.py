import abc
from abc import ABC
from typing import Generic, Iterator

from gflownet.api.env_base import TAction, TActionSpace, TState
from gflownet.api.trajectories import Trajectories


class ReplayBufferBase(ABC, Generic[TState, TActionSpace, TAction]):
    """
    A base class for replay buffers. A replay buffer stores terminal states or trajectories and can sample them
    in backward direction using the provided sampler.

    Type parameters:
        TState: The type of the states.
        TActionSpace: The type of the action spaces.
        TAction: The type of the actions.
    """

    @abc.abstractmethod
    def add_trajectories(self, trajectories: Trajectories[TState, TActionSpace, TAction]):
        """
        Add trajectories to the replay buffer.

        Args:
            trajectories: trajectories to add.

        Returns:
            None
        """
        ...

    @abc.abstractmethod
    def get_trajectories_iterator(
        self, n_total_trajectories: int, batch_size: int
    ) -> Iterator[Trajectories]:
        """
        Get an iterator that samples trajectories from the environment. It can be used to sampled trajectories in
            batched manner.
        Args:
            n_total_trajectories: total number of trajectories to sample.
            batch_size: the size of the batch. If -1, the batch size is equal to the number of n_total_trajectories.

        Returns:
            an iterator that samples trajectories.
        """
        ...

    @abc.abstractmethod
    def state_dict(self) -> dict:
        """
        Return the state of the replay buffer as a dictionary.

        Returns:
            a dictionary containing the state of the replay buffer.
        """
        ...

    @abc.abstractmethod
    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load the state of the replay buffer from a dictionary.

        Args:
            state_dict: a dictionary containing the state of the replay buffer.

        Returns:
            None
        """
        ...

    @abc.abstractmethod
    def set_device(self, device: str):
        """
        Set the device on which to perform the computations.

        Args:
            device: a device to set.

        Returns:
            None
        """
        ...

    @abc.abstractmethod
    def clear_sampling_cache(self):
        """
        Clear the sampling cache of the replay buffer and underlying objects (e.g. samplers with policies). Some
            objects may use caching to speed up the sampling process.

        Returns:
            None
        """
        ...

    @abc.abstractmethod
    def clear_action_embedding_cache(self):
        """
        Clear the action embedding cache of the replay buffer and underlying objects (e.g. samplers with policies). Some
           policies may embed and cache the actions.

        Returns:
            None
        """
        ...
