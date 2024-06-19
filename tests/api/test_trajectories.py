from typing import List

import torch

from gflownet.api.trajectories import Trajectories


def test__trajectories__add_source_states():
    trajectories: Trajectories[int, List[int], int] = Trajectories()

    trajectories.add_source_states([1, 2, 3])
    assert trajectories.get_last_states_flat() == [1, 2, 3]


def test__trajectories__add_actions_states_single():
    trajectories: Trajectories[int, List[int], int] = Trajectories()

    trajectories.add_source_states([1, 2, 3])
    trajectories.add_actions_states(
        actions=[0, 0, 0],
        states=[4, 4, 4],
        forward_action_spaces=[[0], [0], [0]],
        backward_action_spaces=[[0], [0], [0]],
    )

    assert trajectories.get_actions_flat() == [0, 0, 0]
    assert trajectories.get_forward_action_spaces_flat() == [[0], [0], [0]]
    assert trajectories.get_backward_action_spaces_flat() == [[0], [0], [0]]
    assert trajectories.get_last_states_flat() == [4, 4, 4]


def test__trajectories__add_actions_states_double():
    trajectories: Trajectories[int, List[int], int] = Trajectories()

    trajectories.add_source_states([1, 2, 3])
    trajectories.add_actions_states(
        actions=[0, 0, 0],
        states=[4, 4, 4],
        forward_action_spaces=[[0], [0], [0]],
        backward_action_spaces=[[0], [0], [0]],
    )
    trajectories.add_actions_states(
        actions=[1, 1, 1],
        states=[5, 5, 5],
        forward_action_spaces=[[1], [1], [1]],
        backward_action_spaces=[[0], [0], [1]],
    )

    assert trajectories.get_actions_flat() == [0, 1, 0, 1, 0, 1]
    assert trajectories.get_forward_action_spaces_flat() == [[0], [1], [0], [1], [0], [1]]
    assert trajectories.get_backward_action_spaces_flat() == [[0], [0], [0], [0], [0], [1]]
    assert trajectories.get_last_states_flat() == [5, 5, 5]
    assert trajectories.get_non_last_states_flat() == [1, 4, 2, 4, 3, 4]
    assert trajectories.get_non_source_states_flat() == [4, 5, 4, 5, 4, 5]


def test__trajectories__add_actions_states_double_with_mask():
    trajectories: Trajectories[int, List[int], int] = Trajectories()

    trajectories.add_source_states([1, 2, 3])
    trajectories.add_actions_states(
        actions=[0, 0, 0],
        states=[4, 4, 4],
        forward_action_spaces=[[0], [0], [0]],
        backward_action_spaces=[[0], [0], [0]],
    )
    trajectories.add_actions_states(
        actions=[1, 1],
        states=[5, 5],
        forward_action_spaces=[[1], [1]],
        backward_action_spaces=[[1], [1]],
        not_terminated_mask=[True, True, False],
    )

    assert torch.equal(trajectories.get_index_flat(), torch.tensor([0, 0, 1, 1, 2]))
    assert trajectories.get_actions_flat() == [0, 1, 0, 1, 0]
    assert trajectories.get_forward_action_spaces_flat() == [[0], [1], [0], [1], [0]]
    assert trajectories.get_backward_action_spaces_flat() == [[0], [1], [0], [1], [0]]
    assert trajectories.get_last_states_flat() == [5, 5, 4]
    assert trajectories.get_non_last_states_flat() == [1, 4, 2, 4, 3]
    assert trajectories.get_non_source_states_flat() == [4, 5, 4, 5, 4]


def test__trajectories__reversed():
    trajectories: Trajectories[int, int, int] = Trajectories()

    trajectories.add_source_states([1, 2, 3])
    trajectories.add_actions_states(
        actions=[0, 0, 0],
        states=[4, 4, 4],
        forward_action_spaces=[0, 0, 0],
        backward_action_spaces=[0, 2, 0],
    )
    trajectories.add_actions_states(
        actions=[1, 1],
        states=[5, 5],
        forward_action_spaces=[1, 1],
        backward_action_spaces=[1, 1],
        not_terminated_mask=[True, True, False],
    )

    assert set(trajectories.get_forward_action_spaces_flat()) != set(
        trajectories.reversed().get_forward_action_spaces_flat()
    )
    assert set(trajectories.get_backward_action_spaces_flat()) != set(
        trajectories.reversed().get_backward_action_spaces_flat()
    )
    assert trajectories != trajectories.reversed()
    assert trajectories == trajectories.reversed().reversed()
    assert trajectories.reversed() != trajectories.reversed().reversed()


def test__trajectories__masked_select():
    trajectories: Trajectories[int, List[int], int] = Trajectories()

    trajectories.add_source_states([1, 2, 3])
    trajectories.add_actions_states(
        actions=[0, 0, 0],
        states=[4, 4, 4],
        forward_action_spaces=[[0], [0], [0]],
        backward_action_spaces=[[0], [0], [0]],
    )
    trajectories.add_actions_states(
        actions=[1, 1],
        states=[5, 5],
        forward_action_spaces=[[1], [1]],
        backward_action_spaces=[[1], [1]],
        not_terminated_mask=[True, True, False],
    )

    n_actions = len(trajectories.get_actions_flat())
    forward_log_probs = torch.arange(n_actions).float()
    backward_log_probs = torch.arange(n_actions).float()
    trajectories.set_forward_log_probs_flat(forward_log_probs)
    trajectories.set_backward_log_probs_flat(backward_log_probs)

    mask = torch.tensor([True, False, True]).bool()
    trajectories = trajectories.masked_select(mask)
    assert trajectories.get_actions_flat() == [0, 1, 0]
    assert trajectories.get_forward_action_spaces_flat() == [[0], [1], [0]]
    assert trajectories.get_backward_action_spaces_flat() == [[0], [1], [0]]
    assert trajectories.get_last_states_flat() == [5, 4]
    assert torch.equal(trajectories.get_forward_log_probs_flat(), torch.tensor([0, 1, 4]).float())
    assert torch.equal(trajectories.get_backward_log_probs_flat(), torch.tensor([0, 1, 4]).float())


def test__trajectories_from_trajectories():
    t1 = Trajectories()
    t1._states_list = [[0, 0, 0, 0], [1, 1, 1]]
    t1._forward_action_spaces_list = [[0, 0, 0], [1, 1]]
    t1._backward_action_spaces_list = [[0, 0, 0], [1, 1]]
    t1._actions_list = [[0, 0, 0], [1, 1]]
    t1._forward_log_probs_flat = torch.tensor([0, 0, 0, 1, 1])
    t1._backward_log_probs_flat = torch.tensor([0, 0, 0, 1, 1])

    t2 = Trajectories()
    t2._states_list = [[2, 2, 2], [3, 3]]
    t2._forward_action_spaces_list = [[2, 2], [3]]
    t2._backward_action_spaces_list = [[2, 2], [3]]
    t2._actions_list = [[2, 2], [3]]
    t2._forward_log_probs_flat = torch.tensor([2, 2, 3])
    t2._backward_log_probs_flat = torch.tensor([2, 2, 3])

    trajectories = Trajectories.from_trajectories([t1, t2])
    assert trajectories.get_actions_flat() == [0, 0, 0, 1, 1, 2, 2, 3]
    assert trajectories.get_forward_action_spaces_flat() == [0, 0, 0, 1, 1, 2, 2, 3]
    assert trajectories.get_backward_action_spaces_flat() == [0, 0, 0, 1, 1, 2, 2, 3]
    assert trajectories.get_last_states_flat() == [0, 1, 2, 3]
    assert torch.equal(
        trajectories.get_forward_log_probs_flat(), torch.tensor([0, 0, 0, 1, 1, 2, 2, 3])
    )
    assert torch.equal(
        trajectories.get_backward_log_probs_flat(), torch.tensor([0, 0, 0, 1, 1, 2, 2, 3])
    )
    assert torch.equal(trajectories.get_index_flat(), torch.tensor([0, 0, 0, 1, 1, 2, 2, 3]))


def test__trajectories_scatter_add():
    trajectories = Trajectories()
    trajectories._states_list = [[0, 0, 0, 0], [1, 1, 1]]
    trajectories._forward_action_spaces_list = [[0, 0, 0], [1, 1]]
    trajectories._backward_action_spaces_list = [[0, 0, 0], [1, 1]]
    trajectories._actions_list = [[0, 0, 0], [1, 1]]
    trajectories._forward_log_probs_flat = torch.tensor([0, 0, 0, 1, 1]).float()
    trajectories._backward_log_probs_flat = torch.tensor([0, 0, 0, 1, 1]).float()

    index = trajectories.get_index_flat()
    log_probs = trajectories.get_forward_log_probs_flat()
    trajectories_log_probs = torch.zeros(len(trajectories), dtype=torch.float)
    trajectories_log_probs = torch.scatter_add(
        input=trajectories_log_probs, index=index, src=log_probs, dim=0
    )
    assert torch.equal(trajectories_log_probs, torch.tensor([0, 2]).float())
