from typing import List

import pytest
import torch
from dgllife.utils import WeaveAtomFeaturizer

from gflownet.gfns.retro.api.data_structures import Molecule
from gflownet.gfns.retro.policy.gnns import EmbeddingGNN


@pytest.mark.parametrize("smiles_list", [["CCN"], ["CCN", "CCO"], ["CCN", "CCO", "CCOC"]])
def test__embedding_gnn__returns_different_embeddings(smiles_list: List[str]):
    gnn = EmbeddingGNN(
        hidden_dim=32,
        num_attention_heads=4,
        num_layers=3,
        node_featurizer=WeaveAtomFeaturizer(),
    )
    molecules_list = [Molecule(smiles) for smiles in smiles_list]
    output = gnn.forward(molecules_list)
    max_num_atoms = max([mol.rdkit_mol.GetNumAtoms() for mol in molecules_list])
    assert output.shape == (len(smiles_list), max_num_atoms, gnn.hidden_dim)
    assert torch.isnan(output).sum() == 0
    assert torch.isinf(output).sum() == 0
    for i in range(1, len(smiles_list)):
        assert not torch.allclose(output[i], output[0])
