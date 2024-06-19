import copy
import random
from functools import singledispatch
from typing import Any, Callable, Dict, FrozenSet, List

import gin
from rdkit import Chem
from rdkit.Chem import ChiralType, rdChemReactions

from gflownet.api.env_base import EnvBase
from gflownet.gfns.retro.api.data_structures import Bag, SortedList, get_symbol
from gflownet.gfns.retro.api.retro_api import (
    EarlyTerminalRetroState,
    EarlyTerminateRetroAction,
    EarlyTerminateRetroActionSpace,
    FirstPhaseRetroAction,
    FirstPhaseRetroActionSpace,
    FirstPhaseRetroState,
    MappingTuple,
    Molecule,
    RetroAction,
    RetroActionSpace,
    RetroState,
    SecondPhaseRetroAction,
    SecondPhaseRetroActionSpace,
    SecondPhaseRetroState,
    TerminalRetroState,
    ThirdPhaseRetroAction,
    ThirdPhaseRetroActionSpace,
    ThirdPhaseRetroState,
)
from gflownet.gfns.retro.api.retro_data_factory import RetroDataFactory
from gflownet.gfns.retro.retro_utils import chiral_type_map_inv, get_backward_template


@gin.configurable()
class RetroEnv(EnvBase[RetroState, RetroActionSpace, RetroAction]):
    """
    The RetroGFN environment. It provides the action spaces and the transitions for three phases of the template
    composition process of RetroGFN.
    """

    def __init__(self, data_factory: RetroDataFactory, optimize_for_inference: bool = False):
        """
        Initialize the RetroGFN environment.

        Args:
            data_factory: a factory that provides data for the RetroGFN training or inference.
            optimize_for_inference: whether to cache the states and action spaces for the inference.
        """
        super().__init__()
        self.source_states = data_factory.get_source_states()
        self.product_patterns = data_factory.get_product_patterns()
        self.reactant_patterns = data_factory.get_reactant_patterns()
        self.terminal_states = data_factory.get_terminal_states()

        self.product_to_possible_first_phase_actions: Dict[
            Molecule, FrozenSet[FirstPhaseRetroAction]
        ] = {}

        self.product_pattern_to_reactant_patterns_indices = {}
        for product_pattern in self.product_patterns:
            reactant_patterns_indices = [
                i
                for i, reactant_pattern in enumerate(self.reactant_patterns)
                if reactant_pattern.mappable_symbols.is_subset(product_pattern.mappable_symbols)
            ]
            self.product_pattern_to_reactant_patterns_indices[
                product_pattern
            ] = reactant_patterns_indices

        # The above dicts should probably be replaced with @singledispatch,
        # but it's not clear whether it will be more readable
        self.state_type_to_forward_action_space_fn: Dict[Any, Callable] = {
            FirstPhaseRetroState: self._get_forward_first_phase_action_space,
            SecondPhaseRetroState: self._get_forward_second_phase_action_space,
            ThirdPhaseRetroState: self._get_forward_third_phase_action_space,
        }
        self.state_type_to_backward_action_space_fn: Dict[Any, Callable] = {
            SecondPhaseRetroState: self._get_backward_second_phase_action_space,
            ThirdPhaseRetroState: self._get_backward_third_phase_action_space,
            TerminalRetroState: self._get_backward_terminal_action_space,
            EarlyTerminalRetroState: self._get_backward_early_terminal_action_space,
        }
        self.action_type_to_forward_apply_fn: Dict[Any, Callable] = {
            FirstPhaseRetroAction: self._apply_forward_first_phase_action,
            SecondPhaseRetroAction: self._apply_forward_second_phase_action,
            ThirdPhaseRetroAction: self._apply_forward_third_phase_action,
            EarlyTerminateRetroAction: self._apply_forward_early_terminate_action,
        }
        self.action_type_to_backward_apply_fn: Dict[Any, Callable] = {
            FirstPhaseRetroAction: self._apply_backward_first_phase_action,
            SecondPhaseRetroAction: self._apply_backward_second_phase_action,
            ThirdPhaseRetroAction: self._apply_backward_third_phase_action,
            EarlyTerminateRetroAction: self._apply_backward_early_terminate_action,
        }

        self.optimize_for_inference = optimize_for_inference
        self.state_3_to_terminal: Dict[ThirdPhaseRetroState, TerminalRetroState] = {}
        self.state_to_action_space_cache: Dict[RetroState, RetroActionSpace] = {}

    def reset_inference_cache(self):
        """
        Reset the cache of the states and action spaces for the inference.
        """
        self.state_3_to_terminal = {}
        self.state_to_action_space_cache = {}

    def get_forward_action_spaces(self, states: List[RetroState]) -> List[RetroActionSpace]:
        action_spaces: List[RetroActionSpace] = []
        for state in states:
            if self.optimize_for_inference and state in self.state_to_action_space_cache:
                action_space = self.state_to_action_space_cache[state]
            else:
                action_space = self.state_type_to_forward_action_space_fn[type(state)](state)
                if self.optimize_for_inference:
                    self.state_to_action_space_cache[state] = action_space
            if action_space.is_empty():
                action_space = EarlyTerminateRetroActionSpace()
            action_spaces.append(action_space)
        return action_spaces

    def _get_forward_first_phase_action_space(
        self, state: FirstPhaseRetroState
    ) -> FirstPhaseRetroActionSpace:
        """
        Get the action space for the first phase of the retrosynthesis process.
        """
        if state.product not in self.product_to_possible_first_phase_actions:
            actions_list: List[FirstPhaseRetroAction] = []
            for lhs_pattern in self.product_patterns:
                matches = state.product.rdkit_mol.GetSubstructMatches(
                    lhs_pattern.rdkit_mol, uniquify=False
                )
                actions_list.extend(
                    FirstPhaseRetroAction(subgraph_idx=match, product_pattern=lhs_pattern)
                    for match in matches
                )
            actions = frozenset(actions_list)
            self.product_to_possible_first_phase_actions[state.product] = actions
            return FirstPhaseRetroActionSpace(possible_actions=actions)
        else:
            actions = self.product_to_possible_first_phase_actions[state.product]
            return FirstPhaseRetroActionSpace(possible_actions=actions)

    def _get_forward_second_phase_action_space(
        self, state: SecondPhaseRetroState
    ) -> SecondPhaseRetroActionSpace:
        """
        Get the action space for the second phase of the retrosynthesis process.
        """
        reactant_patterns_indices = self.product_pattern_to_reactant_patterns_indices[
            state.product_pattern
        ]
        reactants_mappable = Bag.from_bags(r.mappable_symbols for r in state.reactant_patterns)
        actions_mask = [False] * len(self.reactant_patterns)
        product_mappable = state.product_pattern.mappable_symbols
        for i in reactant_patterns_indices:
            reactant_pattern = self.reactant_patterns[i]
            sum_mappable = reactants_mappable | reactant_pattern.mappable_symbols
            if sum_mappable.is_subset(product_mappable):
                actions_mask[i] = True

        return SecondPhaseRetroActionSpace(actions_mask=tuple(actions_mask))

    def _get_forward_third_phase_action_space(
        self, state: ThirdPhaseRetroState
    ) -> ThirdPhaseRetroActionSpace:
        """
        Get the action space for the third phase of the retrosynthesis process.
        """
        matched_product_nodes = set(mapping.product_node for mapping in state.atom_mapping)
        matched_reactant_tuples = set(
            (mapping.reactant, mapping.reactant_node) for mapping in state.atom_mapping
        )
        possible_tuples = []
        for symbol, product_pattern_indices in state.product_pattern.symbol_to_indices.items():
            for reactant_pattern_idx, reactant_pattern in enumerate(state.reactant_patterns):
                reactant_pattern_indices = reactant_pattern.symbol_to_indices[symbol]
                for p_idx in product_pattern_indices:
                    if p_idx not in matched_product_nodes:
                        for r_idx in reactant_pattern_indices:
                            reactant_tuple = (reactant_pattern_idx, r_idx)
                            if reactant_tuple not in matched_reactant_tuples:
                                possible_tuples.append(
                                    MappingTuple(
                                        product_node=p_idx,
                                        reactant=reactant_pattern_idx,
                                        reactant_node=r_idx,
                                    )
                                )

        return ThirdPhaseRetroActionSpace(possible_actions=frozenset(possible_tuples))

    def get_backward_action_spaces(self, states: List[RetroState]) -> List[RetroActionSpace]:
        action_spaces: List[RetroActionSpace] = []
        for state in states:
            action_space = self.state_type_to_backward_action_space_fn[type(state)](state)
            action_spaces.append(action_space)
        return action_spaces

    def _get_backward_second_phase_action_space(
        self, state: SecondPhaseRetroState
    ) -> SecondPhaseRetroActionSpace | FirstPhaseRetroActionSpace:
        """
        Get the action space for the second phase of the retrosynthesis process.
        """
        if len(state.reactant_patterns) == 0:
            possible_action = FirstPhaseRetroAction(
                subgraph_idx=state.subgraph_idx,
                product_pattern=state.product_pattern,
            )
            return FirstPhaseRetroActionSpace(possible_actions=frozenset([possible_action]))
        else:
            actions_mask = [False] * len(self.reactant_patterns)
            for pattern in state.reactant_patterns:
                actions_mask[self.reactant_patterns.index(pattern)] = True
            return SecondPhaseRetroActionSpace(actions_mask=tuple(actions_mask))

    def _get_backward_third_phase_action_space(
        self, state: ThirdPhaseRetroState
    ) -> ThirdPhaseRetroActionSpace | SecondPhaseRetroActionSpace:
        """
        Get the action space for the third phase of the retrosynthesis process.
        """
        if len(state.atom_mapping) == 0:
            previous_state = SecondPhaseRetroState(
                product=state.product,
                product_pattern=state.product_pattern,
                subgraph_idx=state.subgraph_idx,
                reactant_patterns=Bag(state.reactant_patterns),
            )
            return self._get_backward_second_phase_action_space(previous_state)  # type: ignore
        else:
            return ThirdPhaseRetroActionSpace(state.atom_mapping)

    def _get_backward_terminal_action_space(self, state: TerminalRetroState) -> RetroActionSpace:
        """
        Get the action space for the terminal state of the retrosynthesis process.
        """
        previous_state = ThirdPhaseRetroState(
            product=state.product,
            product_pattern=state.product_pattern,
            subgraph_idx=state.subgraph_idx,
            reactant_patterns=state.reactant_patterns,
            atom_mapping=state.atom_mapping,
        )
        return self._get_backward_third_phase_action_space(previous_state)

    def _get_backward_early_terminal_action_space(
        self, state: EarlyTerminalRetroState
    ) -> RetroActionSpace:
        """
        Get the action space for the early terminal state of the retrosynthesis process.
        """
        return EarlyTerminateRetroActionSpace()

    def apply_forward_actions(
        self, states: List[RetroState], actions: List[RetroAction]
    ) -> List[RetroState]:
        new_states: List[RetroState] = []
        for state, action in zip(states, actions):
            new_state = self.action_type_to_forward_apply_fn[type(action)](state, action)
            new_states.append(new_state)
        return new_states

    def _apply_forward_first_phase_action(
        self, state: FirstPhaseRetroState, action: FirstPhaseRetroAction
    ) -> SecondPhaseRetroState:
        """
        Apply the action of the first phase of the retrosynthesis process.
        """
        return SecondPhaseRetroState(
            product=state.product,
            subgraph_idx=action.subgraph_idx,
            product_pattern=action.product_pattern,
            reactant_patterns=Bag(),
        )

    def _apply_forward_second_phase_action(
        self, state: SecondPhaseRetroState, action: SecondPhaseRetroAction
    ) -> SecondPhaseRetroState | ThirdPhaseRetroState:
        """
        Apply the action of the second phase of the retrosynthesis process.
        """
        new_reactants_patterns = state.reactant_patterns | {
            self.reactant_patterns[action.reactant_pattern_idx]
        }
        reactants_mappable = Bag.from_bags(r.mappable_symbols for r in new_reactants_patterns)
        if reactants_mappable == state.product_pattern.mappable_symbols:
            return ThirdPhaseRetroState(
                product=state.product,
                subgraph_idx=state.subgraph_idx,
                product_pattern=state.product_pattern,
                reactant_patterns=SortedList(new_reactants_patterns),
                atom_mapping=frozenset(),
            )
        else:
            return SecondPhaseRetroState(
                product=state.product,
                subgraph_idx=state.subgraph_idx,
                product_pattern=state.product_pattern,
                reactant_patterns=new_reactants_patterns,
            )

    def _apply_forward_third_phase_action(
        self, state: ThirdPhaseRetroState, action: ThirdPhaseRetroAction
    ) -> ThirdPhaseRetroState | TerminalRetroState:
        """
        Apply the action of the third phase of the retrosynthesis process.
        """
        new_atom_mapping = state.atom_mapping | {action.mapping}
        new_state = ThirdPhaseRetroState(
            product=state.product,
            subgraph_idx=state.subgraph_idx,
            product_pattern=state.product_pattern,
            reactant_patterns=state.reactant_patterns,
            atom_mapping=new_atom_mapping,
        )
        if len(new_atom_mapping) == state.product_pattern.rdkit_mol.GetNumAtoms():
            if self.optimize_for_inference and new_state in self.state_3_to_terminal:
                return self.state_3_to_terminal[new_state]
            else:
                terminal_state = self._get_terminal_state(new_state)
                if self.optimize_for_inference:
                    self.state_3_to_terminal[new_state] = terminal_state
                return terminal_state
        else:
            return new_state

    def _apply_forward_early_terminate_action(
        self, state: RetroState, action: EarlyTerminateRetroAction
    ) -> EarlyTerminalRetroState:
        """
        Apply the action of the early termination of the retrosynthesis process.
        """
        return EarlyTerminalRetroState(previous_state=state)

    def _get_terminal_state(self, state: ThirdPhaseRetroState) -> TerminalRetroState:
        """
        Get the terminal state of the retrosynthesis process from the third phase state.
        """
        template_dict = get_backward_template(
            state.product_pattern, state.reactant_patterns, state.atom_mapping
        )

        def _matches(reactants: List[Chem.Mol]) -> bool:
            """
            Check if the reactants match the product pattern and the atom mapping. The rdkit applies the reaction
            template to all possible places in the product molecule, resulting in a set of sets of reactants.
            In RetroGFN, we already pre-matched the template in a concrete place in the product molecule, so we need to
            retrieve the template application that corresponds to our pre-matched template.
            Args:
                reactants: a list of reactants that are generated by application of the reaction template to the product.

            Returns:
                True if the reactants were obtained by application of our pre-matched template, False otherwise.
            """
            for reactant in reactants:
                for atom in reactant.GetAtoms():
                    if atom.HasProp("old_mapno"):
                        product_pattern_idx = atom.GetIntProp("old_mapno") - 1
                        product_idx = atom.GetIntProp("react_atom_idx")
                        if state.subgraph_idx[product_pattern_idx] != product_idx:
                            return False
            return True

        def _fix_reactant(
            reactant_mol: Chem.Mol,
            product_mol: Chem.Mol,
            map_num_to_reactant_pattern_node: Dict[int, Chem.Atom],
        ) -> Chem.Mol:
            """
            Apply the relative changes of number of hydrogens, charge, and chiral tag to the reactants' atoms.
            Args:
                reactant_mol: output reactant molecule.
                product_mol: input product molecule.
                map_num_to_reactant_pattern_node: a mapping that helps to find the corresponding reactant pattern node.

            Returns:
                The reactant molecule with the updated atom properties.
            """
            for atom in reactant_mol.GetAtoms():
                if not atom.IsInRing() and atom.GetIsAromatic():
                    atom.SetIsAromatic(False)

                if atom.HasProp("old_mapno"):
                    map_num = atom.GetIntProp("old_mapno")
                    reactant_pattern_node = map_num_to_reactant_pattern_node[map_num]
                    product_atom = product_mol.GetAtomWithIdx(atom.GetIntProp("react_atom_idx"))
                    assert (
                        get_symbol(product_atom)
                        == get_symbol(atom)
                        == get_symbol(reactant_pattern_node)
                    )
                    charge = product_atom.GetFormalCharge() + reactant_pattern_node.GetIntProp(
                        "charge_diff"
                    )
                    hydrogens = product_atom.GetTotalNumHs() + reactant_pattern_node.GetIntProp(
                        "hydrogen_diff"
                    )
                    chiral = chiral_type_map_inv[reactant_pattern_node.GetIntProp("chiral_diff")]
                    if chiral != ChiralType.CHI_UNSPECIFIED:
                        atom.SetChiralTag(chiral)
                    atom.SetFormalCharge(charge)
                    atom.SetNumExplicitHs(hydrogens)

            for bond in reactant_mol.GetBonds():
                if not bond.IsInRing():
                    bond.SetIsAromatic(False)
                    if str(bond.GetBondType()) == "AROMATIC":
                        bond.SetBondType(Chem.rdchem.BondType.SINGLE)

            return reactant_mol

        try:
            reaction = rdChemReactions.ReactionFromSmarts(template_dict["template"])
            product_mol = copy.deepcopy(state.product.rdkit_mol)
            reactants_list = reaction.RunReactants([product_mol])
            reactants_list = [reactants for reactants in reactants_list if _matches(reactants)]
            assert len(reactants_list) == 1
            reactants = reactants_list[0]
            mapped_reactant_pattern_nodes = [
                node
                for pattern in template_dict["reactant_pattern_mols"]
                for node in pattern.GetAtoms()
                if node.GetAtomMapNum() != 0
            ]
            map_num_to_reactant_pattern_node = {
                node.GetAtomMapNum(): node for node in mapped_reactant_pattern_nodes
            }
            reactant_mols = [
                _fix_reactant(reactant_mol, product_mol, map_num_to_reactant_pattern_node)
                for reactant_mol in reactants
            ]
            reactants_smiles = [Chem.MolToSmiles(reactant) for reactant in reactant_mols]
            reactants_smiles = [smiles for smiles in reactants_smiles if smiles is not None]
            reactants = Bag(Molecule(smiles) for smiles in reactants_smiles)
            valid = len(reactants) > 0
        except Exception as e:
            reactants = Bag()
            valid = False

        return TerminalRetroState(
            product=state.product,
            reactants=reactants,
            subgraph_idx=state.subgraph_idx,
            product_pattern=state.product_pattern,
            reactant_patterns=state.reactant_patterns,
            atom_mapping=state.atom_mapping,
            template=template_dict["template"],
            valid=valid,
        )

    def apply_backward_actions(
        self, states: List[RetroState], actions: List[RetroAction]
    ) -> List[RetroState]:
        new_states: List[RetroState] = []
        for state, action in zip(states, actions):
            new_state = self.action_type_to_backward_apply_fn[type(action)](state, action)
            new_states.append(new_state)
        return new_states

    def _apply_backward_first_phase_action(
        self, state: SecondPhaseRetroState, action: FirstPhaseRetroAction
    ) -> FirstPhaseRetroState:
        """
        Apply the action of the first phase of the retrosynthesis process.
        """
        return FirstPhaseRetroState(
            product=state.product,
        )

    def _apply_backward_second_phase_action(
        self, state: SecondPhaseRetroState | ThirdPhaseRetroState, action: SecondPhaseRetroAction
    ) -> SecondPhaseRetroState:
        """
        Apply the action of the second phase of the retrosynthesis process.
        """
        pattern_to_remove = self.reactant_patterns[action.reactant_pattern_idx]
        reactant_patterns_list = list(state.reactant_patterns)
        reactant_patterns_list.remove(pattern_to_remove)
        reactants_patterns = Bag(reactant_patterns_list)

        return SecondPhaseRetroState(
            product=state.product,
            subgraph_idx=state.subgraph_idx,
            product_pattern=state.product_pattern,
            reactant_patterns=reactants_patterns,
        )

    def _apply_backward_third_phase_action(
        self, state: ThirdPhaseRetroState | TerminalRetroState, action: ThirdPhaseRetroAction
    ) -> ThirdPhaseRetroState:
        """
        Apply the action of the third phase of the retrosynthesis process.
        """
        new_atom_mapping = state.atom_mapping - {action.mapping}
        return ThirdPhaseRetroState(
            product=state.product,
            subgraph_idx=state.subgraph_idx,
            product_pattern=state.product_pattern,
            reactant_patterns=state.reactant_patterns,
            atom_mapping=new_atom_mapping,
        )

    def _apply_backward_early_terminate_action(
        self, state: EarlyTerminalRetroState, action: EarlyTerminateRetroAction
    ) -> ThirdPhaseRetroState:
        """
        Apply the action of the early termination of the retrosynthesis process.
        """
        return state.previous_state

    def get_terminal_mask(self, states: List[RetroState]) -> List[bool]:
        return [
            isinstance(state, (TerminalRetroState, EarlyTerminalRetroState)) for state in states
        ]

    def sample_source_states(self, n_states: int) -> List[RetroState]:
        return random.choices(self.source_states, k=n_states)

    def get_source_mask(self, states: List[RetroState]) -> List[bool]:
        return [isinstance(state, FirstPhaseRetroState) for state in states]

    def sample_terminal_states(self, n_states: int) -> List[RetroState]:
        return random.choices(self.terminal_states, k=n_states)

    def get_num_source_states(self) -> int:
        return len(self.source_states)

    def get_source_states_at_index(self, index: List[int]) -> List[RetroState]:
        return [self.source_states[i] for i in index]

    def get_num_terminal_states(self) -> int:
        return len(self.terminal_states)

    def get_terminal_states_at_index(self, index: List[int]) -> List[RetroState]:
        return [self.terminal_states[i] for i in index]
