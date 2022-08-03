#!/usr/bin/env python3
import datetime
import os

import hydra
from cyy_naive_lib.log import set_file_handler
from cyy_torch_algorithm.hydra.hydra_config import HyDRAConfig

config = HyDRAConfig()


@hydra.main(config_path="conf", version_base=None)
def load_config(conf):
    global config
    if len(conf) == 1:
        conf = next(iter(conf.values()))
    HyDRAConfig.load_config(config, conf, check_config=True)


if __name__ == "__main__":
    load_config()
    set_file_handler(
        os.path.join(
            "log",
            "hydra",
            config.dc_config.dataset_name,
            config.model_config.model_name,
            "{date:%Y-%m-%d_%H:%M:%S}.log".format(date=datetime.datetime.now()),
        )
    )
    hydra_trainer = config.create_trainer()
    hydra_trainer.train()
