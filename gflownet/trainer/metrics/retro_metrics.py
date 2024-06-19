from typing import Any, Dict, List, Sequence, Tuple

import gin
import numpy as np
import torch
from rdkit import DataStructs
from rdkit.Chem import AllChem, MolFromSmiles
from torchtyping import TensorType

from gflownet.api.trajectories import Trajectories
from gflownet.trainer.metrics.metric_base import MetricsBase


@gin.configurable()
class RetroTopKAccuracy(MetricsBase):
    def __init__(self, n_repeats: int, k_list: Sequence[int] = (1, 5, 10)):
        super().__init__()
        self.n_repeats = n_repeats
        self.k_list = k_list

    def compute_metrics(self, trajectories: Trajectories) -> Dict[str, float]:
        proxy_components = trajectories.get_reward_outputs().proxy_components
        assert proxy_components is not None
        dataset_output = proxy_components["dataset"].detach().cpu()
        log_probs = trajectories.get_forward_log_probs_flat().detach().cpu()
        index = trajectories.get_index_flat()
        trajectories_log_probs = torch.zeros_like(dataset_output)
        trajectories_log_probs = torch.scatter_add(
            input=trajectories_log_probs, index=index, src=log_probs, dim=0
        )

        proxy_output = dataset_output.view(-1, self.n_repeats)
        trajectories_log_probs = trajectories_log_probs.view(-1, self.n_repeats)
        trajectories_log_probs, sorted_indices = torch.sort(
            trajectories_log_probs, descending=True, dim=1
        )
        proxy_output = torch.gather(proxy_output, dim=1, index=sorted_indices)
        results = (proxy_output.cumsum(dim=1) > 0).long()

        n_results = len(results)
        tok_k_results = {
            f"top_{k}_acc": results[:, k - 1].sum().item() / n_results for k in self.k_list
        }

        indices = torch.arange(1, results.shape[1] + 1).unsqueeze(0).expand_as(results)
        ranks = (results * indices).float()
        ranks = torch.masked_fill(ranks, ranks == 0, float("inf"))
        ranks = torch.min(ranks, dim=1).values
        mrr_results = {"mrr": (1 / ranks).mean().item()}

        return tok_k_results | mrr_results
