#!/usr/bin/env python3
import datetime
import os

from cyy_naive_lib.log import set_file_handler

from config import get_config

if __name__ == "__main__":
    config = get_config()
    set_file_handler(
        os.path.join(
            "log",
            "hypergradient",
            config.dataset_name,
            config.model_name,
            "{date:%Y-%m-%d_%H:%M:%S}.log".format(date=datetime.datetime.now()),
        )
    )
    hyper_gradient_trainer = config.create_trainer()
    hyper_gradient_trainer.train()
