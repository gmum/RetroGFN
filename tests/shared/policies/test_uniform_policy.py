from dataclasses import dataclass
from typing import List

import pytest

from gflownet.shared.policies.uniform_policy import (
    IndexedActionSpaceBase,
    UniformPolicy,
)
from gflownet.utils.helpers import seed_everything


@dataclass
class IndexedList(IndexedActionSpaceBase):
    actions: List[int]

    def get_action_at_idx(self, idx: int) -> int:
        return self.actions[idx]

    def get_idx_of_action(self, action: int) -> int:
        return self.actions.index(action)

    def get_possible_actions_indices(self) -> List[int]:
        return list(range(len(self.actions)))


@pytest.mark.repeat(10)
def test__uniform_policy__returns_only_possible_actions():
    policy = UniformPolicy()
    states = [0, 1, 2, 3]
    action_spaces = [
        IndexedList([0, 1]),
        IndexedList([0, 1, 2]),
        IndexedList([0, 1]),
        IndexedList([0, 1]),
    ]
    actions = policy.sample_actions(states, action_spaces)
    for action, action_space in zip(actions, action_spaces):
        assert action in action_space.actions


def test__uniform_policy__covers_all_possible_actions():
    seed_everything(42)
    policy = UniformPolicy()
    states = [0, 1, 2, 3]
    action_spaces = [IndexedList([0, 1, 2, 3])]
    all_sampled_actions = set()
    for _ in range(10):
        actions = policy.sample_actions(states, action_spaces)
        all_sampled_actions.update(actions)
    assert all_sampled_actions == set(action_spaces[0].actions)


def test__uniform_policy__is_uniform():
    n_repeats = 1000
    n_actions = 4
    n_repeats_per_action = n_repeats // n_actions
    seed_everything(42)
    policy = UniformPolicy()
    states = [0, 1, 2, 3]
    action_spaces = [IndexedList([0, 1, 2, 3])]
    actions_count = {action: 0 for action in action_spaces[0].actions}
    for _ in range(n_repeats):
        actions = policy.sample_actions(states, action_spaces)
        for action in actions:
            actions_count[action] += 1
    mean_count = sum(actions_count.values()) / len(actions_count)
    assert abs(mean_count - n_repeats_per_action) < 0.1
    assert max(actions_count.values()) - min(actions_count.values()) < 0.1 * n_repeats_per_action
