import sys
from pathlib import Path
from typing import List, Sequence

import gin
from torch import nn

from gflownet import ROOT_DIR
from gflownet.gfns.retro.api.retro_api import TerminalRetroState
from gflownet.gfns.retro.api.retro_data_factory import RetroDataFactory
from gflownet.gfns.retro.proxies.retro_feasibility_proxy_base import (
    RetroFeasibilityProxyBase,
)


class ReactionChemformer(nn.Module):
    def __init__(
        self,
        model_dir: Path | str,
        num_results: int = 1,
    ):
        super().__init__()
        sys.path.append(str(ROOT_DIR / "external/syntheseus"))
        sys.path.append(str(ROOT_DIR / "external/syntheseus/external"))
        from external.syntheseus.syntheseus.reaction_prediction.inference.chemformer import (
            ChemformerModel,
        )

        self.model = ChemformerModel(
            model_dir=model_dir,
            is_forward=True,
        )
        self.num_results = num_results

    def forward(self, reactants: List[List[str]], products: List[str]) -> List[float]:
        from syntheseus import Reaction

        from external.syntheseus.syntheseus.interface.bag import Bag
        from external.syntheseus.syntheseus.interface.molecule import Molecule

        reactants_list = [Bag(Molecule(s) for s in r) for r in reactants]
        products_list = [Molecule(s) for s in products]

        output: List[Sequence[Reaction]] = self.model(reactants_list, num_results=self.num_results)
        probs = []
        assert len(output) == len(products_list)
        for reaction_list, product in zip(output, products_list):
            prob = 0.0
            for reaction in reaction_list:
                predicted_smiles_set = set(m.smiles for m in reaction.products)
                if product.smiles in predicted_smiles_set:
                    prob += reaction.metadata["probability"]
            probs.append(prob)

        return probs


@gin.configurable()
class RetroChemformerProxy(RetroFeasibilityProxyBase):
    def __init__(
        self,
        checkpoint_path: str | Path,
        data_factory: RetroDataFactory,
        quantized: bool = False,
        dataset_lambda: float = 1.0,
        max_score: float = 1.0,
        num_results: int = 1,
    ):
        super().__init__(
            data_factory=data_factory,
            quantized=quantized,
            dataset_lambda=dataset_lambda,
            max_score=max_score,
        )
        self.model = ReactionChemformer(
            model_dir=checkpoint_path,
            num_results=num_results,
        )

    def set_device(self, device: str):
        super().set_device(device)
        self.model.model.model.to(device)
        self.model.model.device = device

    def _compute_value_with_model(self, states: List[TerminalRetroState]) -> List[float]:
        reactants_smiles = [[m.smiles for m in state.reactants] for state in states]
        products = [state.product.smiles for state in states]

        return self.model(reactants_smiles, products)
