from itertools import compress
from pathlib import Path
from typing import List

import pytest

from gflownet.gfns.retro import RetroDataFactory, RetroEnv

from .fixtures import (
    path_to_product_patterns,
    path_to_reactant_patterns,
    path_to_truncated_train,
    retro_data_factory,
    retro_env,
)


def test__retro_data_factory__get_products(retro_data_factory: RetroDataFactory):
    products = retro_data_factory.get_products()
    for product in products[1:]:
        assert product != products[0]


def test__retro_data_factory__is_deterministic(
    path_to_product_patterns: Path, path_to_reactant_patterns: Path, path_to_truncated_train: Path
):
    data_factory_1 = RetroDataFactory(
        split_path=path_to_truncated_train,
        product_patterns_path=path_to_product_patterns,
        reactant_patterns_path=path_to_reactant_patterns,
    )
    data_factory_2 = RetroDataFactory(
        split_path=path_to_truncated_train,
        product_patterns_path=path_to_product_patterns,
        reactant_patterns_path=path_to_reactant_patterns,
    )
    assert data_factory_1.get_products() == data_factory_2.get_products()
    assert data_factory_1.get_reactant_patterns() == data_factory_2.get_reactant_patterns()


def test__retro_data_factory__get_reactant_patterns(retro_data_factory: RetroDataFactory):
    patterns = retro_data_factory.get_reactant_patterns()
    for pattern in patterns[1:]:
        assert pattern != patterns[0]


def test__retro_data_factory__get_product_patterns(retro_data_factory: RetroDataFactory):
    patterns = retro_data_factory.get_product_patterns()
    for pattern in patterns[1:]:
        assert pattern != patterns[0]


def test__retro_data_factory__get_terminal_states(retro_data_factory: RetroDataFactory):
    terminal_states = retro_data_factory.get_terminal_states()
    for terminal_state in terminal_states[1:]:
        assert terminal_state != terminal_states[0]


def test__retro_data_factory__get_terminal_states__sensible(
    retro_data_factory: RetroDataFactory, retro_env: RetroEnv
):
    terminal_states = retro_data_factory.get_terminal_states()
    backward_action_spaces = retro_env.get_backward_action_spaces(terminal_states)
    actions = [
        action_space.get_action_at_idx(action_space.get_possible_actions_indices()[0])
        for action_space in backward_action_spaces
    ]

    previous_states = retro_env.apply_backward_actions(terminal_states, actions)
    retrieved_states = retro_env.apply_forward_actions(previous_states, actions)

    assert retrieved_states == terminal_states
