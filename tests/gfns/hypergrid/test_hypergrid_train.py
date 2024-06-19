from pathlib import Path

from gfns.helpers.training_test_helpers import helper__test_training__runs_properly


def test__hypergrid_train(tmp_path: Path):
    config_path = "configs/hypergrid.gin"
    config_override_str = ""
    helper__test_training__runs_properly(config_path, config_override_str, tmp_path)
