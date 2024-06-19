import gin

from gflownet.api.sampler_base import SamplerBase
from gflownet.api.trajectories import Trajectories
from gflownet.gfns.retro.api.data_structures import Reaction
from gflownet.gfns.retro.api.retro_api import (
    RetroAction,
    RetroActionSpace,
    RetroState,
    TerminalRetroState,
)
from gflownet.gfns.retro.api.retro_data_factory import RetroDataFactory
from gflownet.shared.replay_buffers.reward_prioritized_replay_buffer import (
    RewardPrioritizedReplayBuffer,
)


@gin.configurable()
class RetroReplayBuffer(RewardPrioritizedReplayBuffer[RetroState, RetroActionSpace, RetroAction]):
    """
    Reward prioritized replay buffer for the retro environment that additionally filter out reactions that are
    in the dataset.
    """

    def __init__(
        self,
        sampler: SamplerBase[RetroState, RetroActionSpace, RetroAction],
        data_factory: RetroDataFactory,
        max_size: int = int(1e6),
        temperature: float = 1.0,
    ):
        super().__init__(sampler=sampler, max_size=max_size, temperature=temperature)
        self.dataset_reactions = data_factory.get_reactions()

    def add_trajectories(
        self, trajectories: Trajectories[RetroState, RetroActionSpace, RetroAction]
    ):
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
            if isinstance(state, TerminalRetroState):
                reaction = Reaction(product=state.product, reactants=state.reactants)
                if reaction not in self.dataset_reactions:
                    self._add_state(state, proxy_value.item())
