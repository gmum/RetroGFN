import copy
from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar

TState = TypeVar("TState")
TActionSpace = TypeVar("TActionSpace")
TAction = TypeVar("TAction")


class EnvBase(Generic[TState, TActionSpace, TAction], ABC):
    """
    Base class for environments. It provides a minimal and flexible interface that can be used to implement
    environments with dynamic action spaces. The reward is decoupled from the environment, so that environment should
    only describe the possible transitions between states. The environment can be reversed to enable backward
    sampling of the trajectories.

    Type Parameters:
        TState: The type of the states.
        TActionSpace: The type of the action spaces. Action space describes the possible actions that can be taken in a
            given state.
        TAction: The type of the actions.

    Attributes:
        is_reversed: A boolean that indicates if the environment is reversed.
    """

    def __init__(self):
        self.is_reversed = False

    @abstractmethod
    def get_forward_action_spaces(self, states: List[TState]) -> List[TActionSpace]:
        """
        Get the forward action spaces for the given states.
        Args:
            states: a list of states of length `n_states`.

        Returns:
            a list of forward action spaces corresponding to the given states of length `n_states`. Forward action space
            describes the possible actions that can be taken in a given state in the forward direction.
        """
        ...

    @abstractmethod
    def get_backward_action_spaces(self, states: List[TState]) -> List[TActionSpace]:
        """
        Get the backward action spaces for the given states.
        Args:
            states: a list of states of length `n_states`.

        Returns:
            a list of backward action spaces corresponding to the given states of length `n_states`. Backward action
            space describes the possible actions that can be taken in a given state in the backward direction.
        """
        ...

    @abstractmethod
    def apply_forward_actions(self, states: List[TState], actions: List[TAction]) -> List[TState]:
        """
        Apply the forward actions to the given states.
        Args:
            states: a list of states of length `n_states`.
            actions: a list of actions of length `n_states`.

        Returns:
            a list of states after applying the forward actions of length `n_states`.
        """
        ...

    @abstractmethod
    def apply_backward_actions(self, states: List[TState], actions: List[TAction]) -> List[TState]:
        """
        Apply the backward actions to the given states.
        Args:
            states: a list of states of length `n_states`.
            actions: a list of actions of length `n_states`.

        Returns:
            a list of states after applying the backward actions of length `n_states`.
        """
        ...

    @abstractmethod
    def get_source_mask(self, states: List[TState]) -> List[bool]:
        """
        Get the mask indicating which states are source states (the states from which no backward action can be
            performed).
        Args:
            states: a list of states of length `n_states`.

        Returns:
            a list of boolean values of length `n_states` indicating which states are source states.
        """
        ...

    @abstractmethod
    def get_terminal_mask(self, states: List[TState]) -> List[bool]:
        """
        Get the mask indicating which states are terminal states (the states from which no forward action can be
            performed).
        Args:
            states: a list of states of length `n_states`.

        Returns:
            a list of boolean values of length `n_states` indicating which states are terminal states.
        """
        ...

    @abstractmethod
    def sample_source_states(self, n_states: int) -> List[TState]:
        """
        Sample source states. Source states are the states from which no backward action can be performed.

        Args:
            n_states: the number of source states to sample.

        Returns:
            a list of source states of length `n_states`.
        """
        ...

    @abstractmethod
    def sample_terminal_states(self, n_states: int) -> List[TState]:
        """
        Sample terminal states. Terminal states are the states from which no forward action can be performed. It is
            used when the environment is reversed.

        Args:
            n_states: the number of terminal states to sample.

        Returns:
            a list of terminal states of length `n_states`.
        """
        ...

    @abstractmethod
    def get_num_source_states(self) -> int:
        """
        Get the number of source states. It is usually set to 0. It is only used in `SequentialSampler`.

        Returns:
            The number of source states.
        """
        ...

    @abstractmethod
    def get_num_terminal_states(self) -> int:
        """
        Get the number of terminal states. It is only used in `SequentialSampler` when the environment is reversed.

        Returns:
            The number of terminal states.
        """
        ...

    @abstractmethod
    def get_source_states_at_index(self, index: List[int]) -> List[TState]:
        """
        Get the source states at the given index. It is only used in `SequentialSampler`.

        Args:
            index: the list of indices of the source states.

        Returns:
            a list of source states at the given index.
        """
        ...

    @abstractmethod
    def get_terminal_states_at_index(self, index: List[int]) -> List[TState]:
        """
        Get the terminal states at the given index. It is only used in `SequentialSampler` when the environment is
            reversed.

        Args:
            index: the list of indices of the terminal states.

        Returns:
            a list of terminal states at the given index.
        """
        ...

    def reversed(self) -> "EnvBase[TState, TActionSpace, TAction]":
        """
        Get the reversed environment. It swaps the forward and backward action spaces, forward and backward actions,
            terminal and source masks, and source and terminal states, etc., so that the environment can be used in the
            backward direction for trajectory sampling.

        Returns:
            the reversed environment.
        """
        env = copy.deepcopy(self)
        env.is_reversed = not env.is_reversed
        env.get_forward_action_spaces, env.get_backward_action_spaces = (  # type: ignore
            env.get_backward_action_spaces,
            env.get_forward_action_spaces,
        )
        env.apply_forward_actions, env.apply_backward_actions = (  # type: ignore
            env.apply_backward_actions,
            env.apply_forward_actions,
        )
        env.get_terminal_mask, env.get_source_mask = env.get_source_mask, env.get_terminal_mask  # type: ignore
        env.sample_source_states, env.sample_terminal_states = (  # type: ignore
            env.sample_terminal_states,
            env.sample_source_states,
        )
        env.get_source_states_at_index, env.get_terminal_states_at_index = (  # type: ignore
            env.get_terminal_states_at_index,
            env.get_source_states_at_index,
        )
        env.get_num_source_states, env.get_num_terminal_states = (  # type: ignore
            env.get_num_terminal_states,
            env.get_num_source_states,
        )
        return env
