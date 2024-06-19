from typing import List

import gin

from gflownet.api.env_base import EnvBase, TState

from .hypergrid_api import HyperGridAction, HyperGridActionSpace, HyperGridState


@gin.configurable()
class HyperGridEnv(EnvBase[HyperGridState, HyperGridActionSpace, HyperGridAction]):
    """
    A simple environment that represents a hyper-grid world with a fixed size and a fixed number of dimensions.
    The agent can move in each dimension by `max_num_steps` step at a time. The agent can also terminate the episode.
    """

    def __init__(self, size: int, n_dimensions: int, max_num_steps: int):
        """

        Args:
            size: the size of the hyper-grid world.
            n_dimensions: the number of dimensions of the hyper-grid world.
            max_num_steps: maximum number of steps the agent can take in each dimension. The action space will contain
            all possible combinations of steps in each dimension up to `max_num_steps`.
        """
        super().__init__()
        self.size = size
        self.n_dimensions = n_dimensions
        all_actions = [HyperGridAction(steps=(0,) * n_dimensions, terminate=True, idx=0)]
        prev_actions_layer = [HyperGridAction(steps=(0,) * n_dimensions, terminate=True, idx=0)]
        for i in range(1, max_num_steps + 1):
            new_actions = []
            for action in prev_actions_layer:
                for j in range(n_dimensions):
                    new_steps = list(action.steps)
                    new_steps[j] += 1
                    new_action = HyperGridAction(steps=tuple(new_steps), idx=len(all_actions))
                    new_actions.append(new_action)
                    all_actions.append(new_action)
            prev_actions_layer = new_actions
        self.all_actions = tuple(all_actions)
        assert all(i == action.idx for i, action in enumerate(self.all_actions))

    @property
    def num_actions(self) -> int:
        """
        Returns:
            the number of actions in the action space.
        """
        return len(self.all_actions)

    def get_forward_action_spaces(self, states: List[HyperGridState]) -> List[HyperGridActionSpace]:
        action_spaces = []
        for state in states:
            actions_mask = [False] * len(self.all_actions)
            for i, action in enumerate(self.all_actions):
                new_state = self._apply_forward_action(state, action)
                actions_mask[i] = new_state.valid
            action_spaces.append(
                HyperGridActionSpace(
                    all_actions=self.all_actions, possible_actions_mask=tuple(actions_mask)
                )
            )
        return action_spaces

    def get_backward_action_spaces(
        self, states: List[HyperGridState]
    ) -> List[HyperGridActionSpace]:
        action_spaces = []
        for state in states:
            actions_mask = [False] * len(self.all_actions)
            for i, action in enumerate(self.all_actions):
                new_state = self._apply_backward_action(state, action)
                actions_mask[i] = new_state.valid
            action_spaces.append(
                HyperGridActionSpace(
                    all_actions=self.all_actions, possible_actions_mask=tuple(actions_mask)
                )
            )
        return action_spaces

    def apply_forward_actions(
        self, states: List[HyperGridState], actions: List[HyperGridAction]
    ) -> List[HyperGridState]:
        return [self._apply_forward_action(state, action) for state, action in zip(states, actions)]

    def apply_backward_actions(
        self, states: List[HyperGridState], actions: List[HyperGridAction]
    ) -> List[HyperGridState]:
        return [
            self._apply_backward_action(state, action) for state, action in zip(states, actions)
        ]

    def _apply_forward_action(
        self, state: HyperGridState, action: HyperGridAction
    ) -> HyperGridState:
        """
        Apply the given forward action to the given state.

        Args:
            state: a state.
            action: a forward action.

        Returns:
            a new state after applying the action.
        """
        if action.terminate:
            return HyperGridState(coords=state.coords, terminal=True, valid=True)
        new_coords = tuple(c + s for c, s in zip(state.coords, action.steps))
        valid = all(0 <= c < self.size for c in new_coords)
        return HyperGridState(coords=new_coords, terminal=False, valid=valid)

    def _apply_backward_action(
        self, state: HyperGridState, action: HyperGridAction
    ) -> HyperGridState:
        """
        Apply the given backward action to the given state.

        Args:
            state: a state.
            action: a backward action.

        Returns:
            a new state after applying the action.
        """
        if state.terminal and not action.terminate:
            return HyperGridState(coords=state.coords, terminal=True, valid=False)
        if action.terminate:
            return HyperGridState(coords=state.coords, terminal=False, valid=state.terminal)
        new_coords = tuple(c - s for c, s in zip(state.coords, action.steps))
        valid = all(0 <= c < self.size for c in new_coords)
        return HyperGridState(coords=new_coords, terminal=False, valid=valid)

    def get_terminal_mask(self, states: List[HyperGridState]) -> List[bool]:
        return [state.terminal for state in states]

    def get_source_mask(self, states: List[HyperGridState]) -> List[bool]:
        return [all(c == 0 for c in state.coords) for state in states]

    def sample_source_states(self, n_states: int) -> List[HyperGridState]:
        source_state = HyperGridState(coords=(0,) * self.n_dimensions, terminal=False, valid=True)
        return [source_state] * n_states

    def sample_terminal_states(self, n_states: int) -> List[HyperGridState]:
        raise NotImplementedError()

    def get_num_source_states(self) -> int:
        raise NotImplementedError()

    def get_source_states_at_index(self, index: List[int]) -> List[TState]:
        raise NotImplementedError()

    def get_num_terminal_states(self) -> int:
        raise NotImplementedError()

    def get_terminal_states_at_index(self, index: List[int]) -> List[TState]:
        raise NotImplementedError()
