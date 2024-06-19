import copy
from typing import Any, Dict, Iterable

import torch
from rdkit import Chem
from rdkit.Chem import ChiralType
from torch import Tensor
from torchtyping import TensorType

from gflownet.gfns.retro.api.data_structures import Pattern, ReactantPattern, SortedList
from gflownet.gfns.retro.api.retro_api import MappingTuple

chiral_type_map = {
    ChiralType.CHI_UNSPECIFIED: 0,
    ChiralType.CHI_TETRAHEDRAL_CW: -1,
    ChiralType.CHI_TETRAHEDRAL_CCW: 1,
}
chiral_type_map_inv = {v: k for k, v in chiral_type_map.items()}


def pad_and_concatenate(x: Tensor, y: Tensor) -> Tensor:
    """
    A helper function to pad the tensors along the second dimension and concatenate them along the first dimension.

    Args:
        x: a tensor of shape [b1, l1, hidden_size]
        y: a tensor of shape [b2, l2, hidden_size]

    Returns:
        A tensor of shape [b1 + b2, max(l1, l2), hidden_size]
    """
    if x.shape[1] > y.shape[1]:
        y = torch.cat(
            [
                y,
                torch.zeros(
                    [y.shape[0], x.shape[1] - y.shape[1], y.shape[2]],
                    dtype=y.dtype,
                    device=y.device,
                ),
            ],
            dim=1,
        )
    elif x.shape[1] < y.shape[1]:
        x = torch.cat(
            [
                x,
                torch.zeros(
                    [x.shape[0], y.shape[1] - x.shape[1], x.shape[2]],
                    dtype=x.dtype,
                    device=y.device,
                ),
            ],
            dim=1,
        )

    return torch.cat([x, y], dim=0)


def gumbel_cdf(x: TensorType[float], mean: float = 0.0, beta: float = 1) -> TensorType[float]:
    """
    Gumbel CDF function.
    """
    return torch.exp(-torch.exp(-(x - mean) / beta))


def get_backward_template(
    product_pattern: Pattern,
    reactants_patterns: SortedList[ReactantPattern],
    atom_mapping: Iterable[MappingTuple],
) -> Dict[str, Any]:
    """
    A helper function to get the backward template from the product pattern, reactants patterns and atom mapping.
    """
    product_pattern_mol = copy.deepcopy(product_pattern.rdkit_mol)
    reactant_pattern_mols = [copy.deepcopy(pattern.rdkit_mol) for pattern in reactants_patterns]
    for mapping in atom_mapping:
        idx = mapping.product_node + 1
        product_pattern_mol.GetAtomWithIdx(mapping.product_node).SetAtomMapNum(idx)
        reactant_pattern_mols[mapping.reactant].GetAtomWithIdx(mapping.reactant_node).SetAtomMapNum(
            idx
        )

    product_smarts = Chem.MolToSmarts(product_pattern_mol)
    reactants_smarts_list = [f"({Chem.MolToSmarts(mol)})" for mol in reactant_pattern_mols]
    reactants_smarts = ".".join(reactants_smarts_list)
    template = f"({product_smarts})>>{reactants_smarts}"
    return {
        "template": template,
        "product_pattern_mol": product_pattern_mol,
        "reactant_pattern_mols": reactant_pattern_mols,
    }
