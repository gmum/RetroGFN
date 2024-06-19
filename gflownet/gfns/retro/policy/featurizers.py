from typing import Any, Dict, List

import torch
from dgl import random_walk_pe
from dgllife.utils import (
    BaseAtomFeaturizer,
    ConcatFeaturizer,
    WeaveAtomFeaturizer,
    atom_degree_one_hot,
    atom_is_aromatic,
    atom_type_one_hot,
    one_hot_encoding,
)
from dgllife.utils.mol_to_graph import construct_bigraph_from_mol
from rdkit import Chem

ATOM_TYPES = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "Si",
    "P",
    "Cl",
    "Br",
    "Mg",
    "Na",
    "Ca",
    "Fe",
    "As",
    "Al",
    "I",
    "B",
    "V",
    "K",
    "Tl",
    "Yb",
    "Sb",
    "Sn",
    "Ag",
    "Pd",
    "Co",
    "Se",
    "Ti",
    "Zn",
    "H",
    "Li",
    "Ge",
    "Cu",
    "Au",
    "Ni",
    "Cd",
    "In",
    "Mn",
    "Zr",
    "Cr",
    "Pt",
    "Hg",
    "Pb",
    "W",
    "Ru",
    "Nb",
    "Re",
    "Te",
    "Rh",
    "Ta",
    "Tc",
    "Ba",
    "Bi",
    "Hf",
    "Mo",
    "U",
    "Sm",
    "Os",
    "Ir",
    "Ce",
    "Gd",
    "Ga",
    "Cs",
]


def charge_diff_one_hot(atom: Chem.Atom, encode_unknown: bool = False) -> List[bool]:
    """
    Encode the charge difference in the reactant pattern as one-hot encoding.
    """
    allowable_set = list(range(-3, 3))
    charges_diff = atom.GetIntProp("charges_diff") if atom.HasProp("charges_diff") else 0
    return one_hot_encoding(charges_diff, allowable_set, encode_unknown)


def hydrogen_diff_one_hot(atom: Chem.Atom, encode_unknown: bool = False):
    """
    Encode the hydrogen number difference in the reactant pattern as one-hot encoding.
    """
    allowable_set = list(range(-3, 3))
    hydrogen_diff = atom.GetIntProp("hydrogen_diff") if atom.HasProp("hydrogen_diff") else 0
    return one_hot_encoding(hydrogen_diff, allowable_set, encode_unknown)


def chiral_diff_one_hot(atom: Chem.Atom, encode_unknown: bool = False):
    """
    Encode the chiral difference in the reactant pattern as one-hot encoding.
    """

    allowable_set = list(range(-1, 1))
    chiral_diff = atom.GetIntProp("chiral_diff") if atom.HasProp("chiral_diff") else 0
    return one_hot_encoding(chiral_diff, allowable_set, encode_unknown)


def mappable_one_hot(atom: Chem.Atom, encode_unknown: bool = False):
    """
    Encode whether the atom is a mappable atom.
    """
    allowable_set = [0, 1]
    return one_hot_encoding(atom.GetAtomMapNum(), allowable_set, encode_unknown)


class ReactantNodeFeaturizer(BaseAtomFeaturizer):
    """
    Featurizer for the reactant patterns' nodes.
    """

    def __init__(self, atom_data_field="h"):
        super().__init__(
            featurizer_funcs={
                atom_data_field: ConcatFeaturizer(
                    [
                        atom_type_one_hot,
                        atom_degree_one_hot,
                        atom_is_aromatic,
                        charge_diff_one_hot,
                        hydrogen_diff_one_hot,
                        chiral_diff_one_hot,
                        mappable_one_hot,
                    ]
                )
            }
        )


class JointFeaturizer:
    """
    Featurizer that concatenates the atom features and the positional encoding featurizer.
    """

    def __init__(
        self, atom_featurizer: BaseAtomFeaturizer | WeaveAtomFeaturizer, pe_featurizer: Any
    ):
        """
        Initialize the featurizer.
        Args:
            atom_featurizer: the atom level featurizer.
            pe_featurizer: the positional encoding featurizer.
        """
        self.atom_featurizer = atom_featurizer
        self.pe_featurizer = pe_featurizer

    def feat_size(self) -> int:
        """
        Get the size of the features.
        """
        return self.atom_featurizer.feat_size() + self.pe_featurizer.feat_size()

    def __call__(self, mol: Chem.Mol):
        atom_features = self.atom_featurizer(mol)["h"]
        pe_features = self.pe_featurizer(mol)
        return {"h": torch.cat([atom_features, pe_features], dim=-1)}


class RandomWalkPEFeaturizer:
    """
    Featurizer that performs random walk positional encoding.
    """

    def __init__(self, n_steps: int = 16):
        self.n_steps = n_steps

    def feat_size(self) -> int:
        return self.n_steps

    def __call__(self, mol: Chem.Mol):
        graph = construct_bigraph_from_mol(mol)
        return random_walk_pe(graph, k=self.n_steps)
