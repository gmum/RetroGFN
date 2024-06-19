import abc
from typing import Dict, List

import torch

from gflownet.gfns.retro.api.retro_api import RetroState, TerminalRetroState
from gflownet.gfns.retro.api.retro_data_factory import RetroDataFactory
from gflownet.gfns.retro.proxies.retro_dataset_proxy import RetroDatasetProxy
from gflownet.shared.proxies.cached_proxy import CachedProxyBase


class RetroFeasibilityProxyBase(CachedProxyBase[RetroState], abc.ABC):
    def __init__(
        self,
        data_factory: RetroDataFactory,
        quantized: bool = False,
        dataset_lambda: float = 1.0,
        max_score: float = 1.0,
    ):
        super().__init__()
        self.dataset_proxy = RetroDatasetProxy(data_factory)
        self.quantized = quantized
        self.dataset_lambda = dataset_lambda
        self.max_score = max_score

    @property
    def is_non_negative(self) -> bool:
        return True

    @property
    def higher_is_better(self) -> bool:
        return True

    @abc.abstractmethod
    def _compute_value_with_model(self, states: List[TerminalRetroState]) -> List[float]:
        ...

    @torch.no_grad()
    def _compute_proxy_output(self, states: List[RetroState]) -> List[Dict[str, float]]:
        dataset_score = self.dataset_proxy.compute_proxy_output(states).value

        feasibility_score = torch.zeros(len(states), dtype=torch.float32, device=self.device)
        feasibility_states_mask = torch.tensor(
            [isinstance(state, TerminalRetroState) and state.valid for state in states],
            dtype=torch.bool,
            device=self.device,
        )

        states_for_model = [state for state, mask in zip(states, feasibility_states_mask) if mask]
        if len(states_for_model) > 0:
            feasibility = self._compute_value_with_model(states_for_model)  # type: ignore
            feasibility_score[feasibility_states_mask] = torch.tensor(
                feasibility, dtype=torch.float32, device=self.device
            )

        value_score = dataset_score * self.dataset_lambda + feasibility_score
        value_score[value_score > self.max_score] = self.max_score
        if self.quantized:
            value_score = torch.masked_fill(value_score, value_score > 0.0, self.max_score)

        value_list = value_score.cpu().numpy()
        feasibility_list = feasibility_score.cpu().numpy()
        dataset_list = dataset_score.cpu().numpy()

        return [
            {
                "value": value,
                "dataset": dataset,
                "feasibility": feasibility,
            }
            for value, dataset, feasibility in zip(value_list, dataset_list, feasibility_list)
        ]

    def set_device(self, device: str):
        super().set_device(device)
        self.dataset_proxy.set_device(device)
