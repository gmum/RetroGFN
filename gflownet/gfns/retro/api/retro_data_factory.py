from copy import copy
from pathlib import Path
from typing import List, Union

import gin
import pandas as pd

from gflownet.gfns.retro.api.data_structures import Bag, Reaction, SortedList
from gflownet.gfns.retro.api.retro_api import (
    FirstPhaseRetroState,
    MappingTuple,
    Molecule,
    Pattern,
    ReactantPattern,
    SecondPhaseRetroState,
    TerminalRetroState,
    ThirdPhaseRetroState,
)
from gflownet.gfns.retro.retro_utils import get_backward_template


@gin.configurable()
class RetroDataFactory:
    """
    A factory that provides data for the RetroGFN training and inference.
    """

    def __init__(
        self,
        split_path: Union[Path, str, None],
        product_patterns_path: Union[Path, str],
        reactant_patterns_path: Union[Path, str],
        additional_source_file: Union[Path, str, None] = None,
        root_dir: str | Path = ".",
    ):
        """
        Initialize the RetroDataFactory.

        Args:
            split_path: a path to the reactions file.
            product_patterns_path: a path to the product patterns file.
            reactant_patterns_path: a path to the reactant patterns file.
            additional_source_file: an additional file with the source molecules.
            root_dir: the root RetroGFN repository.
        """
        root_dir = Path(root_dir)
        self.split_df = (
            pd.read_csv(root_dir / split_path, sep=";") if split_path is not None else None
        )
        self._products_smiles = (
            self.split_df["product"].tolist() if self.split_df is not None else []
        )
        if additional_source_file is not None:
            additional_source_df = pd.read_csv(root_dir / additional_source_file)
            self._products_smiles += additional_source_df["smiles"].tolist()
        self.product_pattern_df = pd.read_csv(root_dir / product_patterns_path)
        self.reactant_pattern_df = pd.read_csv(root_dir / reactant_patterns_path)
        self._products: List[Molecule] | None = None
        self._source_states: List[FirstPhaseRetroState] | None = None
        self._product_patterns: List[Pattern] | None = None
        self._reactant_patterns: List[ReactantPattern] | None = None
        self._terminal_states: List[TerminalRetroState] | None = None
        self._reactions: List[Reaction] | None = None
        self._first_phase_terminal_states: List[SecondPhaseRetroState] | None = None
        self._second_phase_terminal_states: List[ThirdPhaseRetroState] | None = None
        self._third_phase_terminal_states: List[TerminalRetroState] | None = None

    def get_products(self) -> List[Molecule]:
        if self._products is None:
            self._products = (
                [Molecule(smiles) for smiles in self._products_smiles]
                if self.split_df is not None
                else []
            )
        return copy(self._products)

    def get_source_states(self) -> List[FirstPhaseRetroState]:
        if self._source_states is None:
            self._source_states = [
                FirstPhaseRetroState(molecule) for molecule in self.get_products()
            ]
        return copy(self._source_states)

    def get_product_patterns(self) -> List[Pattern]:
        if self._product_patterns is None:
            self._product_patterns = []
            for _, row in self.product_pattern_df.iterrows():
                pattern = Pattern(
                    smarts=row["pattern"],
                    idx=int(row["id"]),
                )
                self._product_patterns.append(pattern)
        return copy(self._product_patterns)

    def get_reactant_patterns(self) -> List[ReactantPattern]:
        if self._reactant_patterns is None:
            self._reactant_patterns = []
            for _, row in self.reactant_pattern_df.iterrows():
                pattern = ReactantPattern(
                    smarts=row["pattern"],
                    charge_diffs=tuple(eval(row["charge_diffs"])),
                    hydrogen_diffs=tuple(eval(row["hydrogen_diffs"])),
                    chiral_diffs=tuple(eval(row["chiral_diffs"])),
                    idx=int(row["id"]),
                )
                self._reactant_patterns.append(pattern)
        return copy(self._reactant_patterns)

    def get_reactions(self):
        if self._reactions is None:
            self._reactions = []
            for _, row in self.split_df.iterrows():
                reactants = Bag(Molecule(smiles) for smiles in row["reactants"].split("."))
                product = Molecule(row["product"])
                self._reactions.append(Reaction(reactants=reactants, product=product))
        return copy(self._reactions)

    def get_terminal_states(self) -> List[TerminalRetroState]:
        if self._terminal_states is None:
            if self.split_df is None:
                self._terminal_states = []
                return copy(self._terminal_states)

            columns = self.split_df.columns
            required_columns = ["subgraph_idx", "product_pattern_idx", "reactant_patterns_idx"]
            if any(c not in columns for c in required_columns):
                self._terminal_states = []
                return copy(self._terminal_states)

            product_patterns = self.get_product_patterns()
            reactant_patterns_list = self.get_reactant_patterns()
            self._terminal_states = []
            for _, row in self.split_df.iterrows():
                product_pattern = product_patterns[row["product_pattern_idx"]]
                reactant_patterns = SortedList(
                    reactant_patterns_list[idx] for idx in eval(row["reactant_patterns_idx"])
                )
                atom_mapping = frozenset(MappingTuple(*x) for x in eval(row["atom_mapping"]))
                template_dict = get_backward_template(
                    product_pattern, reactant_patterns, atom_mapping
                )
                terminal_state = TerminalRetroState(
                    product=Molecule(row["product"]),
                    subgraph_idx=tuple(eval(row["subgraph_idx"])),
                    product_pattern=product_pattern,
                    reactant_patterns=reactant_patterns,
                    atom_mapping=atom_mapping,
                    reactants=Bag(Molecule(smiles) for smiles in row["reactants"].split(".")),
                    template=template_dict["template"],
                    valid=True,
                )
                self._terminal_states.append(terminal_state)
        return copy(self._terminal_states)
