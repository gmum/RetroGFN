from typing import List

import gin
import torch
from torch import nn
from torch.distributions import Categorical
from torchtyping import TensorType

from gflownet.api.policy_base import PolicyBase

from .hypergrid_env import (
    HyperGridAction,
    HyperGridActionSpace,
    HyperGridEnv,
    HyperGridState,
)


@gin.configurable()
class ForwardHyperGridPolicy(
    PolicyBase[HyperGridState, HyperGridActionSpace, HyperGridAction], nn.Module
):
    """
    A hypergrid policy that samples actions from the action space using a learned MLP.
    """

    def __init__(self, env: HyperGridEnv, hidden_dim: int = 32):
        super().__init__()
        self.size = env.size
        self.n_dimensions = env.n_dimensions
        self.num_actions = env.num_actions
        self.hidden_dim = hidden_dim
        self.score_mlp = nn.Sequential(
            nn.Linear(self.n_dimensions, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_actions),
        )
        self.device = "cpu"

    def forward(
        self, states: List[HyperGridState], action_spaces: List[HyperGridActionSpace]
    ) -> TensorType[float]:
        state_encodings = torch.tensor([state.coords for state in states]).float().to(self.device)
        logits = self.score_mlp(state_encodings)  # (batch_size, num_actions)
        mask = torch.tensor([s.possible_actions_mask for s in action_spaces]).bool().to(self.device)
        logits = logits.masked_fill(~mask, float("-inf"))
        return torch.log_softmax(logits, dim=1).float()

    def sample_actions(
        self, states: List[HyperGridState], action_spaces: List[HyperGridActionSpace]
    ) -> List[HyperGridAction]:
        log_probs = self.forward(states, action_spaces)
        probs = torch.exp(log_probs)
        action_indices = Categorical(probs=probs).sample()
        return [
            action_space.get_action_at_idx(idx)
            for action_space, idx in zip(action_spaces, action_indices)
        ]

    def compute_action_log_probs(
        self,
        states: List[HyperGridState],
        action_spaces: List[HyperGridActionSpace],
        actions: List[HyperGridAction],
        shared_embeddings: None = None,
    ) -> TensorType[float]:
        log_probs = self.forward(states, action_spaces)
        return torch.stack([log_probs[i, action.idx] for i, action in enumerate(actions)]).float()

    def set_device(self, device: str):
        self.to(device)
        self.device = device

    def compute_states_log_flow(self, states: List[HyperGridState]) -> TensorType[float]:
        raise NotImplementedError()

    def clear_sampling_cache(self) -> None:
        pass

    def clear_action_embedding_cache(self) -> None:
        pass
