import argparse
from pathlib import Path

import gin

from gflownet.trainer.trainer import Trainer
from gflownet.utils.helpers import seed_everything
from gin_config import get_time_stamp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    args = parser.parse_args()
    seed = args.seed
    config = args.cfg
    checkpoint_path = args.checkpoint_path

    seed_everything(seed)
    config_name = Path(config).stem
    run_name = f"{config_name}/{get_time_stamp()}"
    gin.parse_config_files_and_bindings([config], bindings=[f'run_name="{run_name}"'])
    trainer = Trainer(resume_path=checkpoint_path)
    trainer.logger.log_code("gflownet")
    trainer.logger.log_to_file(gin.operative_config_str(), "operative_config")
    trainer.logger.log_to_file(gin.config_str(), "config")
    trainer.train()
    trainer.close()
