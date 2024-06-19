from pathlib import Path

import pytest

from gflownet import ROOT_DIR
from gflownet.gfns.retro import (
    RetroDataFactory,
    RetroDatasetProxy,
    RetroEnv,
    RetroForwardPolicy,
)
from gflownet.gfns.retro.proxies.retro_chemformer_proxy import (
    ReactionChemformer,
    RetroChemformerProxy,
)
from gflownet.gfns.retro.proxies.retro_rfm_proxy import RetroRFMProxy


@pytest.fixture(scope="module")
def path_to_product_patterns() -> Path:
    return Path(__file__).parent / "../../assets/retro/product_patterns.csv"


@pytest.fixture(scope="module")
def path_to_reactant_patterns() -> Path:
    return Path(__file__).parent / "../../assets/retro/reactant_patterns.csv"


@pytest.fixture(scope="module")
def path_to_truncated_train() -> Path:
    return Path(__file__).parent / "../../assets/retro/truncated_train_10.csv"


@pytest.fixture(scope="module")
def retro_data_factory(
    path_to_product_patterns: Path, path_to_reactant_patterns: Path, path_to_truncated_train: Path
) -> RetroDataFactory:
    return RetroDataFactory(
        split_path=path_to_truncated_train,
        product_patterns_path=path_to_product_patterns,
        reactant_patterns_path=path_to_reactant_patterns,
    )


@pytest.fixture(scope="module")
def retro_env(retro_data_factory: RetroDataFactory) -> RetroEnv:
    return RetroEnv(
        data_factory=retro_data_factory,
    )


@pytest.fixture(scope="module")
def retro_forward_policy(retro_data_factory: RetroDataFactory) -> RetroForwardPolicy:
    return RetroForwardPolicy(data_factory=retro_data_factory)


@pytest.fixture(scope="module")
def retro_dataset_proxy(retro_data_factory: RetroDataFactory):
    return RetroDatasetProxy(data_factory=retro_data_factory)


@pytest.fixture(scope="module")
def chemformer_forward_path() -> Path:
    return ROOT_DIR / "checkpoints" / "feasibility_proxies" / "chemformer" / "eval"


@pytest.fixture(scope="module")
def retro_chemformer_proxy(retro_data_factory: RetroDataFactory, chemformer_forward_path: Path):
    return RetroChemformerProxy(
        data_factory=retro_data_factory, checkpoint_path=chemformer_forward_path
    )


@pytest.fixture(scope="module")
def rfm_train_checkpoint_path() -> Path:
    return ROOT_DIR / "checkpoints" / "feasibility_proxies" / "rfm" / "eval" / "best_reaction.pt"


@pytest.fixture(scope="module")
def retro_rfm_proxy(retro_data_factory: RetroDataFactory, rfm_train_checkpoint_path: Path):
    return RetroRFMProxy(data_factory=retro_data_factory, checkpoint_path=rfm_train_checkpoint_path)


@pytest.fixture(scope="module")
def reaction_chemformer(chemformer_forward_path: Path) -> ReactionChemformer:
    return ReactionChemformer(
        model_dir=chemformer_forward_path,
    )
