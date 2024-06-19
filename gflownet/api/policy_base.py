from abc import ABC, abstractmethod
from typing import Generic, List

from torchtyping import TensorType

from gflownet.api.env_base import TAction, TActionSpace, TState


class PolicyBase(Generic[TState, TActionSpace, TAction], ABC):
    """
    A base class for policies. Given the current batch of states, a policy samples action. It also computes
    the log probabilities when chosen actions and following states are provided.

    Type parameters:
        TState: The type of the states.
        TActionSpace: The type of the action spaces.
        TAction: The type of the actions.
    """

    @abstractmethod
    def sample_actions(
        self, states: List[TState], action_spaces: List[TActionSpace]
    ) -> List[TAction]:
        """
        Sample actions for the given states and action spaces.

        Args:
            states: a list of states of length `n_states`.
            action_spaces: a list of action spaces of length `n_states`. An action space describes the possible actions
                that can be taken in a given state.

        Returns:
            a list of actions of length `n_states`.
        """
        ...

    @abstractmethod
    def compute_action_log_probs(
        self,
        states: List[TState],
        action_spaces: List[TActionSpace],
        actions: List[TAction],
    ) -> TensorType[float]:
        """
        Compute the log probabilities of the given actions take in the given states (and corresponding action spaces).

        Args:
            states: a list of states of length `n_states`.
            action_spaces: a list of action spaces of length `n_states`. An action space describes the possible actions
                that can be taken in a given state.
            actions: a list of actions chosen in the given states of length `n_states`.

        Returns:
            a tensor of log probabilities of shape `(n_states,)`.
        """
        ...

    @abstractmethod
    def compute_states_log_flow(self, states: List[TState]) -> TensorType[float]:
        """
        Compute the log flows log(F(s)) of the given states. It is used in `SubTrajectoryBalanceGFN`/

        Args:
            states: a list of states of length `n_states`.

        Returns:
            a tensor of log flows of shape `(n_states,)`.
        """
        ...

    @abstractmethod
    def set_device(self, device: str):
        """
        Set the device on which to perform the computations.

        Args:
            device: a string representing the device.

        Returns:
            None
        """
        ...

    @abstractmethod
    def clear_sampling_cache(self) -> None:
        """
        Clear the sampling cache. Some policies may use caching to speed up the sampling process.

        Returns:
            None
        """
        ...

    @abstractmethod
    def clear_action_embedding_cache(self) -> None:
        """
        Clear the action embedding cache. Some policies may embed the actions and cache the embeddings to speed up
            the computation of log probabilities.

        Returns:
            None
        """
        ...
