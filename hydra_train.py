#!/usr/bin/env python3
import datetime
import os

import hydra
from cyy_naive_lib.log import set_file_handler
from cyy_torch_algorithm.hydra.hydra_config import HyDRAConfig

global_config = HyDRAConfig()

remain_config = None


@hydra.main(config_path="conf", version_base=None)
def load_config(conf):
    global remain_config
    if len(conf) == 1:
        conf = next(iter(conf.values()))
    remain_config = HyDRAConfig.load_config(global_config, conf, check_config=True)
    assert not remain_config


if __name__ == "__main__":
    load_config()
    config = global_config
    set_file_handler(
        os.path.join(
            "log",
            "hydra",
            config.dc_config.dataset_name,
            config.model_name,
            "{date:%Y-%m-%d_%H:%M:%S}.log".format(date=datetime.datetime.now()),
        )
    )
    hydra_trainer = config.create_trainer()
    hydra_trainer.train()
