#!/usr/bin/env python3
import datetime
import os

from cyy_naive_lib.log import set_file_handler
from cyy_naive_pytorch_lib.algorithm.hydra.args import (
    add_arguments_to_parser, create_hyper_gradient_trainer_from_args)
from cyy_naive_pytorch_lib.arg_parse import get_parsed_args

if __name__ == "__main__":
    parser = add_arguments_to_parser()
    args = get_parsed_args(parser)

    set_file_handler(
        os.path.join(
            "log",
            "hypergradient",
            args.dataset_name,
            args.model_name,
            "{date:%Y-%m-%d_%H:%M:%S}.log".format(date=datetime.datetime.now()),
        )
    )
    hyper_gradient_trainer = create_hyper_gradient_trainer_from_args(args)
    hyper_gradient_trainer.train()
