from dataclasses import dataclass, field
from typing import Any, FrozenSet, List, Tuple

from gflownet.shared.policies.uniform_policy import IndexedActionSpaceBase

from .data_structures import (
    Bag,
    MappingTuple,
    Molecule,
    Pattern,
    ReactantPattern,
    SortedList,
)


@dataclass(frozen=True)
class FirstPhaseRetroState:
    """
    The state of the first phase of the (single-step) retrosynthesis process.

    Attributes:
        product: an initial product to be retrosynthesized.
    """

    product: Molecule


@dataclass(frozen=True)
class FirstPhaseRetroAction:
    """
    The action of the first phase of the retrosynthesis process.

    Attributes:
        subgraph_idx: the indices of the subgraph in the product molecule that are matched to the product pattern.
        product_pattern: the product pattern matched to the subgraph.
    """

    subgraph_idx: Tuple[int, ...]
    product_pattern: Pattern

    def __post_init__(self):
        if self.product_pattern.is_symmetric():
            if self.subgraph_idx[0] > self.subgraph_idx[-1]:
                subgraph_idx = tuple(reversed(self.subgraph_idx))
                object.__setattr__(self, "subgraph_idx", subgraph_idx)


@dataclass(frozen=True)
class FirstPhaseRetroActionSpace(IndexedActionSpaceBase[FirstPhaseRetroAction]):
    """
    The action space of the first phase of the retrosynthesis process.
    """

    possible_actions: FrozenSet[FirstPhaseRetroAction]

    def get_action_at_idx(self, idx: int) -> FirstPhaseRetroAction:
        return list(self.possible_actions)[idx]

    def get_idx_of_action(self, action: FirstPhaseRetroAction) -> int:
        return list(self.possible_actions).index(action)

    def get_possible_actions_indices(self) -> List[int]:
        return list(range(len(self.possible_actions)))


@dataclass(frozen=True)
class SecondPhaseRetroState:
    """
    The state of the second phase of the retrosynthesis process.

    Attributes:
        product: the initial product to be retrosynthesized.
        subgraph_idx: the indices of the subgraph in the product molecule that are matched to the product pattern.
        product_pattern: the product pattern matched to the subgraph.
        reactant_patterns: the bag of reactant patterns that will be used to compose the final backward template.
    """

    product: Molecule
    subgraph_idx: Tuple[int, ...]
    product_pattern: Pattern
    reactant_patterns: Bag[ReactantPattern]


@dataclass(frozen=True)
class SecondPhaseRetroAction:
    """
    The action of the second phase of the retrosynthesis process.

    Attributes:
        reactant_pattern_idx: the index of the reactant pattern chosen.
    """

    reactant_pattern_idx: int


@dataclass(frozen=True)
class SecondPhaseRetroActionSpace(IndexedActionSpaceBase[SecondPhaseRetroAction]):
    """
    The action space of the second phase of the retrosynthesis process.
    """

    actions_mask: Tuple[bool, ...]

    def get_action_at_idx(self, idx: int) -> SecondPhaseRetroAction:
        return SecondPhaseRetroAction(idx)

    def get_idx_of_action(self, action: SecondPhaseRetroAction) -> int:
        return action.reactant_pattern_idx

    def get_possible_actions_indices(self) -> List[int]:
        return [idx for idx, mask in enumerate(self.actions_mask) if mask]


@dataclass(frozen=True)
class ThirdPhaseRetroState:
    """
    The state of the third phase of the retrosynthesis process.

    Attributes:
        product: the initial product to be retrosynthesized.
        subgraph_idx: the indices of the subgraph in the product molecule that are matched to the product pattern.
        product_pattern: the product pattern matched to the subgraph.
        reactant_patterns: the list of reactant patterns that will be used to compose the final backward template.
        atom_mapping: the atom mapping between the product and reactants patterns.
    """

    product: Molecule
    subgraph_idx: Tuple[int, ...]
    product_pattern: Pattern
    reactant_patterns: SortedList[ReactantPattern]
    atom_mapping: FrozenSet[MappingTuple]


@dataclass(frozen=True)
class ThirdPhaseRetroAction:
    """
    The action of the third phase of the retrosynthesis process.

    Attributes:
        mapping: the atom mapping between a single atom of product and reactants patterns.
    """

    mapping: MappingTuple


@dataclass(frozen=True)
class ThirdPhaseRetroActionSpace(IndexedActionSpaceBase[ThirdPhaseRetroAction]):
    """
    The action space of the third phase of the retrosynthesis process.
    """

    possible_actions: FrozenSet[MappingTuple]

    def get_action_at_idx(self, idx: int) -> ThirdPhaseRetroAction:
        return ThirdPhaseRetroAction(list(self.possible_actions)[idx])

    def get_idx_of_action(self, action: ThirdPhaseRetroAction) -> int:
        return list(self.possible_actions).index(action.mapping)

    def get_possible_actions_indices(self) -> List[int]:
        return list(range(len(self.possible_actions)))


@dataclass(frozen=True)
class EarlyTerminateRetroAction:
    """
    The action of the early termination of the retrosynthesis process.
    """

    pass


@dataclass(frozen=True)
class EarlyTerminateRetroActionSpace(IndexedActionSpaceBase[EarlyTerminateRetroAction]):
    """
    The action space of the early termination of the retrosynthesis process.
    """

    def get_action_at_idx(self, idx: int) -> EarlyTerminateRetroAction:
        return EarlyTerminateRetroAction()

    def get_idx_of_action(self, action: EarlyTerminateRetroAction) -> int:
        return 0

    def get_possible_actions_indices(self) -> List[int]:
        return [0]


@dataclass(frozen=True)
class EarlyTerminalRetroState:
    """
    The state of the early termination of the retrosynthesis process. It should never occur during the training,
    but may be useful during the inference.

    Attributes:
        previous_state: the state of the retrosynthesis process that led to the early termination.
    """

    previous_state: Any


@dataclass(frozen=True)
class TerminalRetroState:
    """
    The terminal state of the retrosynthesis process.

    Attributes:
        product: the initial product to be retrosynthesized.
        reactants: the reactants that were used to synthesize the product.
        subgraph_idx: the indices of the subgraph in the product molecule that are matched to the product pattern.
        product_pattern: the product pattern matched to the subgraph.
        reactant_patterns: the list of reactant patterns that were used to compose the final backward template.
        atom_mapping: the atom mapping between the product and reactants patterns.
        template: the backward template that was used to synthesize the product.
        valid: whether the retrosynthesis process was successful.
    """

    product: Molecule = field(hash=True, compare=True)
    reactants: Bag[Molecule] = field(hash=True, compare=True)
    subgraph_idx: Tuple[int, ...] = field(hash=False, compare=False)
    product_pattern: Pattern = field(hash=False, compare=False)
    reactant_patterns: SortedList[ReactantPattern] = field(hash=False, compare=False)
    atom_mapping: FrozenSet[MappingTuple] = field(hash=False, compare=False)
    template: str = field(hash=False, compare=False)
    valid: bool = field(hash=False, compare=False)

    def __repr__(self):
        return (
            f"product={self.product.smiles}\n"
            f"reactants={[r.smiles for r in self.reactants]}\n"
            f"template={self.template}\n"
        )


RetroState = (
    FirstPhaseRetroState
    | SecondPhaseRetroState
    | ThirdPhaseRetroState
    | EarlyTerminalRetroState
    | TerminalRetroState
)

RetroAction = (
    FirstPhaseRetroAction
    | SecondPhaseRetroAction
    | ThirdPhaseRetroAction
    | EarlyTerminateRetroAction
)

RetroActionSpace = (
    FirstPhaseRetroActionSpace
    | SecondPhaseRetroActionSpace
    | ThirdPhaseRetroActionSpace
    | EarlyTerminateRetroActionSpace
)
