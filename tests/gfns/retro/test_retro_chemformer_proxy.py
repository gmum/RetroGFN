import pandas as pd
import pytest
from gfns.helpers.proxy_test_helpers import helper__test_proxy__returns_sensible_values

from .fixtures import *


@pytest.mark.parametrize("n_trajectories", [1, 10])
def test__retro_dataset_proxy__returns_sensible_values(
    retro_chemformer_proxy: RetroChemformerProxy, retro_env: RetroEnv, n_trajectories: int
):
    helper__test_proxy__returns_sensible_values(
        retro_env, retro_chemformer_proxy, n_trajectories=n_trajectories
    )


def test__reaction_chemformer__outputs_backtranslates(
    reaction_chemformer: ReactionChemformer, path_to_truncated_train: Path
):
    df = pd.read_csv(path_to_truncated_train, sep=";")
    products = df["product"].tolist()
    reactants = [reactants.split(".") for reactants in df["reactants"].tolist()]
    outputs = reaction_chemformer.forward(products=products, reactants=reactants)
    assert sum(outputs) > 0
