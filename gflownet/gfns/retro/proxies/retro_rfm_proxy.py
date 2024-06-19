import sys
from pathlib import Path
from typing import List, Literal

import dgl
import gin
import torch
from dgl import DGLGraph
from torchtyping import TensorType
from tqdm import tqdm

from gflownet.api import ROOT_DIR
from gflownet.gfns.retro.api.retro_api import TerminalRetroState
from gflownet.gfns.retro.api.retro_data_factory import RetroDataFactory
from gflownet.gfns.retro.proxies.retro_feasibility_proxy_base import (
    RetroFeasibilityProxyBase,
)
from gflownet.gfns.retro.retro_utils import gumbel_cdf


@gin.configurable()
class RetroRFMProxy(RetroFeasibilityProxyBase):
    def __init__(
        self,
        checkpoint_path: str | Path,
        data_factory: RetroDataFactory,
        quantized: bool = False,
        dataset_lambda: float = 1.0,
        max_score: float = 1.0,
        preprocess_batch_size: int = 16,
        cdf_mode: Literal["sigmoid", "gumbel"] = "gumbel",
    ):
        super().__init__(
            data_factory=data_factory,
            quantized=quantized,
            dataset_lambda=dataset_lambda,
            max_score=max_score,
        )
        assert cdf_mode in ["sigmoid", "gumbel"]
        sys.path.append(str(ROOT_DIR / "external/reaction_feasibility_model"))
        self.cdf_mode = cdf_mode
        with gin.unlock_config():
            from external.reaction_feasibility_model.rfm.models import ReactionGNN
        from external.reaction_feasibility_model.rfm.featurizers import (
            ReactionFeaturizer,
        )

        self.model = ReactionGNN(
            checkpoint_path=checkpoint_path,
        )

        self.data_factory = data_factory
        self.product_embeddings = None
        self.preprocess_batch_size = preprocess_batch_size
        self.featurizer = ReactionFeaturizer()
        self._preprocess()

    def _featurize(self, smiles: str) -> DGLGraph:
        return self.featurizer.featurize_smiles_single(smiles)

    @torch.no_grad()
    def _preprocess(self):
        if self.product_embeddings is not None:
            return
        all_products = self.data_factory.get_products()
        product_graphs = [self._featurize(product.smiles) for product in all_products]
        product_embeddings = []
        for i in tqdm(
            range(0, len(all_products), self.preprocess_batch_size), desc="Preprocessing"
        ):
            batch = product_graphs[i : i + self.preprocess_batch_size]
            batch = dgl.batch(batch).to(self.device)
            embeddings = self.model.product_gnn(batch)
            product_embeddings.append(embeddings)
        self.product_embeddings = torch.cat(product_embeddings, dim=0).cpu()
        self.product_to_index = {product: i for i, product in enumerate(all_products)}

    def set_device(self, device: str):
        super().set_device(device)
        self.model.to(device)

    def _get_reactant_embedding(self, states: List[TerminalRetroState]) -> TensorType[float]:
        reactant_graphs_list = []
        for state in states:
            reactant_graphs = [self._featurize(reactant.smiles) for reactant in state.reactants]
            reactant_graphs_list.append(reactant_graphs)
        reactant_graphs = dgl.batch([dgl.merge(graphs) for graphs in reactant_graphs_list]).to(
            self.device
        )
        return self.model.reactants_gnn(reactant_graphs)

    def _get_product_embeddings(self, states: List[TerminalRetroState]) -> TensorType[float]:
        product_indices = [self.product_to_index[state.product] for state in states]
        product_indices = torch.tensor(product_indices, dtype=torch.long)
        product_embeddings = torch.index_select(self.product_embeddings, 0, product_indices)
        return product_embeddings.to(self.device)

    def _compute_value_with_model(self, states: List[TerminalRetroState]) -> List[float]:
        reactant_embedding = self._get_reactant_embedding(states)
        product_embedding = self._get_product_embeddings(states)
        embedding = self.model.concat(reactant_embedding, product_embedding)
        score = self.model.mlp(embedding).squeeze(-1)

        if self.cdf_mode == "sigmoid":
            return torch.sigmoid(score).cpu().numpy()
        else:
            return gumbel_cdf(score, 0.0, 1.0)
