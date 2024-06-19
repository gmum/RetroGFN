import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Literal

import gin
from torch_geometric import seed_everything

from gflownet.trainer.logger.logger_base import LoggerBase
from gflownet.trainer.trainer import Trainer
from gflownet.utils.helpers import infer_metric_direction
from gin_config import get_time_stamp

BEST_PARAM = "@best_param"


@gin.configurable
def grid_search(
    base_run_name: str,
    base_config_path: str,
    params: List[Dict[str, List[Any]]] | Dict[str, List[Any]],
    logger: LoggerBase,
    best_metric: str = "loss",
    metric_direction: Literal["auto", "min", "max"] = "auto",
    seed: int = 42,
    skip: int = 0,
):
    metric_direction = (
        infer_metric_direction(best_metric) if metric_direction == "auto" else metric_direction
    )
    best_valid_metrics: Dict[str, float] = {}
    best_parameters: Dict[str, Any] = {}

    logger.log_code("gflownet")
    logger.log_to_file(gin.operative_config_str(), "grid_operative_config")
    logger.log_to_file(gin.config_str(), "grid_config")
    logger.log_to_file(json.dumps(params, indent=2), "grid_params")
    logger.close()
    params_list = [params] if isinstance(params, dict) else params
    all_grid_dicts = []
    for param_dict in params_list:
        keys, values = zip(*param_dict.items())
        grid_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
        all_grid_dicts.extend(grid_dicts)

    for idx, grid_dict in enumerate(all_grid_dicts):
        if idx < skip:
            continue
        print(f"Running experiment {idx} with parameters {grid_dict}")
        experiment_name = f"{base_run_name}/params_{idx}"
        bindings = [f'run_name="{experiment_name}"']
        grid_dict = {
            key: (best_parameters[key] if value == BEST_PARAM else value)
            for key, value in grid_dict.items()
        }
        for key, value in grid_dict.items():
            binding = f'{key}="{value}"' if isinstance(value, str) else f"{key}={value}"
            bindings.append(binding)

        config_files = [base_config_path]
        for key, value in grid_dict.items():
            if key.startswith("config_file"):
                config_files.append(value)
        gin.clear_config()
        gin.parse_config_files_and_bindings(config_files, bindings=bindings)
        seed_everything(seed)
        trainer = Trainer()
        trainer.logger.log_code("gflownet")
        trainer.logger.log_to_file("\n".join(bindings), "bindings")
        trainer.logger.log_to_file(gin.operative_config_str(), "operative_config")
        trainer.logger.log_to_file(gin.config_str(), "config")
        trainer.logger.log_config(grid_dict)
        valid_metrics = trainer.train()
        trainer.close()

        if metric_direction == "min":
            is_better = valid_metrics[best_metric] < best_valid_metrics.get(
                best_metric, float("inf")
            )
        else:
            is_better = valid_metrics[best_metric] > best_valid_metrics.get(
                best_metric, float("-inf")
            )
        if is_better:
            best_valid_metrics = valid_metrics
            best_parameters = grid_dict | {"id": f"params_{idx}"}

    json_best_parameters = json.dumps(best_parameters, indent=2)
    json_best_valid_metrics = json.dumps(best_valid_metrics, indent=2)

    logger.restart()
    logger.log_to_file(json_best_parameters, "best_params")
    logger.log_to_file(json_best_valid_metrics, "best_valid_metrics")
    logger.close()

    print(f"Best parameters:\n{json_best_parameters}")
    print(f"Best valid metrics:\n{json_best_valid_metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--skip", type=int, default=0)
    args = parser.parse_args()
    skip = args.skip
    config_path = args.cfg

    config_name = Path(config_path).stem
    run_name = f"{config_name}/{get_time_stamp()}"
    gin.parse_config_files_and_bindings([config_path], bindings=[f'run_name="{run_name}"'])
    grid_search(base_run_name=run_name, base_config_path=config_path, skip=skip)
