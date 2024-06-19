from typing import List

import gin
import torch

from gflownet.api.proxy_base import ProxyBase, ProxyOutput
from gflownet.gfns.retro.api.data_structures import Reaction
from gflownet.gfns.retro.api.retro_api import RetroState, TerminalRetroState
from gflownet.gfns.retro.api.retro_data_factory import RetroDataFactory


@gin.configurable()
class RetroDatasetProxy(ProxyBase[RetroState]):
    def __init__(self, data_factory: RetroDataFactory):
        self.valid_states = set(data_factory.get_reactions())
        self.device = "cpu"

    @property
    def is_non_negative(self) -> bool:
        return True

    @property
    def higher_is_better(self) -> bool:
        return True

    def set_device(self, device: str):
        self.device = device

    def compute_proxy_output(self, states: List[RetroState]) -> ProxyOutput:
        results = []
        for state in states:
            if isinstance(state, TerminalRetroState) and state.valid:
                reaction = Reaction(product=state.product, reactants=state.reactants)
                result = int(reaction in self.valid_states)
            else:
                result = 0
            results.append(result)
        value = torch.tensor(results, dtype=torch.float32, device=self.device)

        return ProxyOutput(value=value, components={"dataset": value})
