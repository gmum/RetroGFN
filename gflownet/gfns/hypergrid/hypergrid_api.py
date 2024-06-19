from dataclasses import dataclass
from typing import List, Tuple

from gflownet.shared.policies.uniform_policy import IndexedActionSpaceBase


@dataclass(frozen=True)
class HyperGridState:
    coords: Tuple[int, ...]
    terminal: bool
    valid: bool

    def __str__(self):
        return f"{self.coords}"


@dataclass(frozen=True)
class HyperGridAction:
    steps: Tuple[int, ...]
    idx: int
    terminate: bool = False

    def __str__(self):
        return str(self.steps)

    def __repr__(self):
        if self.terminate:
            return "Terminate"
        return str(self.steps)


@dataclass(frozen=True)
class HyperGridActionSpace(IndexedActionSpaceBase[HyperGridAction]):
    all_actions: Tuple[HyperGridAction, ...]
    possible_actions_mask: Tuple[bool, ...]

    def get_action_at_idx(self, idx: int) -> HyperGridAction:
        return self.all_actions[idx]

    def get_idx_of_action(self, action: HyperGridAction) -> int:
        return action.idx

    def get_possible_actions_indices(self) -> List[int]:
        return [i for i, mask in enumerate(self.possible_actions_mask) if mask]
