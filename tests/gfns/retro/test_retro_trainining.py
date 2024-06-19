from pathlib import Path

import pytest
from gfns.helpers.training_test_helpers import helper__test_training__runs_properly


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/retro_dataset_proxy.gin",
        "configs/retro_rfm_proxy.gin",
        "configs/retro_chemformer_proxy.gin",
    ],
)
def test__retro__trains_properly(config_path: str, tmp_path: Path):
    config_override_str = (
        "train/RetroDataFactory.split_path='tests/assets/retro/truncated_train_10.csv'\n"
        "valid/RetroDataFactory.split_path='tests/assets/retro/truncated_train_10.csv'"
    )
    helper__test_training__runs_properly(config_path, config_override_str, tmp_path)
