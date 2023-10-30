#!/usr/bin/env python3
import datetime
import os

import hydra
from cyy_naive_lib.log import add_file_handler
from cyy_torch_xai.lean_hydra.lean_hydra_config import LeanHyDRAConfig

config = LeanHyDRAConfig()


@hydra.main(config_path="conf", version_base=None)
def load_config(conf):
    global config
    if len(conf) == 1:
        conf = next(iter(conf.values()))
    LeanHyDRAConfig.load_config(config, conf, check_config=False)


if __name__ == "__main__":
    load_config()
    add_file_handler(
        os.path.join(
            "log",
            config.dc_config.dataset_name,
            config.model_config.model_name,
            "{date:%Y-%m-%d_%H_%M_%S}.log".format(date=datetime.datetime.now()),
        )
    )
    lean_hydra_trainer = config.create_deterministic_trainer()
    lean_hydra_trainer.train()
    lean_hydra_trainer, hook, test_gradient = config.recreate_trainer_and_hook()
    lean_hydra_trainer.train(save_last_model=True)
