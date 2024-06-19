from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union

import gin
import numpy as np
import torch
from more_itertools import chunked

from gflownet.api.trajectories import Trajectories
from gflownet.gfns.retro.api.data_structures import Bag, Molecule, Reaction
from gflownet.gfns.retro.api.retro_api import (
    FirstPhaseRetroState,
    RetroAction,
    RetroActionSpace,
    RetroState,
)
from gflownet.gfns.retro.api.retro_data_factory import RetroDataFactory
from gflownet.gfns.retro.retro_env import RetroEnv
from gflownet.shared.objectives import ConditionedTrajectoryBalanceObjective
from gflownet.shared.samplers.random_sampler import RandomSampler


class RetroGFNSingleStep:
    """
    A wrapper for the RetroGFN model that can be used in external code bases (e.g. Syntheseus).
    """

    def __init__(
        self,
        model_dir: Union[str, Path],
        repeat: int = 10,
        temperature: float = 0.7,
        device: str = "cuda:0",
        root_dir: str = ".",
    ) -> None:
        """
        Initializes the RetroGFN model wrapper.

        Args:
            model_dir: the model directory containing the GFN model checkpoint and config files. It must contain the
                following files:
                - `best_gfn.pt` is the GFN model checkpoint
                - `config.gin` is the GFN model config.
            repeat: how many times to repeat the prediction for each input SMILES (it's used to estimate
                the reaction probability)
            temperature: the temperature to use for the forward policy
            device: a device to perform the inference on
            root_dir: a root directory of the RetroGFN module. It is used to import the necessary classes.
        """
        gin_config_path = Path(model_dir) / "logs/config.gin"
        if not gin_config_path.exists():
            gin_config_path = Path(model_dir) / "logs/config.txt"
        chkpt_path = Path(model_dir) / f"train/checkpoints/best_gfn.pt"

        with gin.unlock_config():
            gin.parse_config_files_and_bindings(
                [str(gin_config_path)], bindings=[f"RetroDataFactory.root_dir='{root_dir}'"]
            )

        data_factory = RetroDataFactory(split_path=None)  # type: ignore
        env = RetroEnv(data_factory=data_factory)  # type: ignore
        env.optimize_for_inference = True

        objective = ConditionedTrajectoryBalanceObjective()  # type: ignore
        chkpt_dict = torch.load(chkpt_path, map_location="cpu")
        objective.load_state_dict(chkpt_dict["model"], strict=True)
        objective.forward_policy.eval()  # type: ignore
        objective.forward_policy.optimize_for_inference = True  # type: ignore
        objective.forward_policy.temperature = temperature  # type: ignore

        self.objective = objective
        self.sampler = RandomSampler(policy=objective.forward_policy, env=env, reward=None)
        self.sampler.set_device(device)
        self.env = env
        self.repeat = repeat

    @torch.no_grad()
    def predict(
        self, products: List[str], num_results: int, batch_size: int = -1
    ) -> List[List[Reaction]]:
        """
        Predicts the reactants for the given products.

        Args:
            products: the list of SMILES strings of the products of length N
            num_results: how many reactions to predict for each product
            batch_size: a batch size to use for the inference. If -1, all computations are performerd in a single batch.

        Returns:
            a list of length N, where each element is a list of reactions for the corresponding product.
        """
        outputs_list: List[List[Reaction]] = []

        for smiles in products:
            self.sampler.clear_action_embedding_cache()
            self.sampler.clear_sampling_cache()
            self.env.reset_inference_cache()
            trajectories_list = []
            source_state = FirstPhaseRetroState(product=Molecule(smiles))
            n_total_repeats = self.repeat * num_results
            batch_size = batch_size if batch_size != -1 else n_total_repeats
            for source_states in chunked([source_state] * n_total_repeats, batch_size):
                trajectories = self.sampler.sample_trajectories_from_sources(source_states)
                terminal_states = trajectories.get_last_states_flat()
                valid_terminal_mask = torch.tensor([t.valid for t in terminal_states])
                trajectories = trajectories.masked_select(valid_terminal_mask)
                if len(trajectories) == 0:
                    continue

                self.objective.assign_log_probs(trajectories)
                trajectories._forward_log_probs_flat = (
                    trajectories._forward_log_probs_flat.detach().cpu()
                )
                trajectories._backward_log_probs_flat = (
                    trajectories._backward_log_probs_flat.detach().cpu()
                )
                trajectories_list.append(trajectories)

            if len(trajectories_list) == 0:
                outputs_list.append([Reaction(reactants=Bag([]), product=Molecule(smiles))])
                continue

            trajectories = Trajectories.from_trajectories(trajectories_list)
            log_probs = trajectories.get_forward_log_probs_flat()
            index = trajectories.get_index_flat()
            trajectories_log_probs = torch.zeros(len(trajectories), dtype=torch.float)
            trajectories_log_probs = torch.scatter_add(
                input=trajectories_log_probs, index=index, src=log_probs, dim=0
            )

            reactants_dict: Dict[Bag[Molecule], float] = {}
            reactants_to_terminal = defaultdict(set)
            for terminal_state, log_prob in zip(
                trajectories.get_last_states_flat(), trajectories_log_probs
            ):
                reactants = terminal_state.reactants
                reactants_to_terminal[reactants].add(terminal_state)
                current = reactants_dict.get(reactants, 0)
                reactants_dict[reactants] = current + np.exp(log_prob)

            predicted_reactants = list(
                sorted(reactants_dict.keys(), key=lambda x: reactants_dict[x], reverse=True)
            )[:num_results]
            outputs = [
                Reaction(
                    reactants=Bag([Molecule(reactant.smiles) for reactant in reactants]),
                    product=Molecule(smiles),
                )
                for reactants in predicted_reactants
            ]
            outputs_list.append(outputs)
        return outputs_list
