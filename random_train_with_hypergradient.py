#!/usr/bin/env python3
import datetime
import json
import os

from cyy_naive_lib.log import set_file_handler
from cyy_naive_pytorch_lib.algorithm.influence_function.args import (
    add_arguments_to_parser, create_hyper_gradient_trainer_from_args)
from cyy_naive_pytorch_lib.arg_parse import get_parsed_args
from cyy_naive_pytorch_lib.dataset import DatasetUtil, replace_dataset_labels
from cyy_naive_pytorch_lib.local_types import MachineLearningPhase
from cyy_naive_pytorch_lib.trainer import Trainer

if __name__ == "__main__":
    parser = add_arguments_to_parser()
    parser.add_argument("--random_percentage", type=float, required=True)
    args = get_parsed_args(parser)
    args.save_dir = os.path.join(args.save_dir, "randomized_model")

    set_file_handler(
        os.path.join(
            "log",
            "randomized_hypergradient",
            args.dataset_name,
            args.model_name,
            "{date:%Y-%m-%d_%H:%M:%S}.log".format(
                date=datetime.datetime.now()),
        )
    )

    hyper_gradient_trainer = create_hyper_gradient_trainer_from_args(args)
    training_dataset = hyper_gradient_trainer.trainer.training_dataset

    randomized_label_map = DatasetUtil(training_dataset).randomize_subset_label(
        args.random_percentage
    )

    os.makedirs(args.save_dir, exist_ok=True)
    with open(
        os.path.join(
            args.save_dir,
            "randomized_label_map.json",
        ),
        mode="wt",
    ) as f:
        json.dump(randomized_label_map, f)
    hyper_gradient_trainer.trainer.set_training_dataset(
        replace_dataset_labels(training_dataset, randomized_label_map)
    )

    def after_epoch_callback(hyper_gradient_trainer: Trainer, epoch, **kwargs):
        trainer = hyper_gradient_trainer.trainer
        test_gradient = trainer.get_inferencer(
            MachineLearningPhase.Test).get_gradient()

        contribution_dict = dict()

        def compute_contribution(sample_index, hyper_gradient):
            sample_index = int(sample_index)
            contribution_dict[sample_index] = (
                -(test_gradient @ hyper_gradient) /
                len(trainer.training_dataset)
            ).data.item()

        hyper_gradient_trainer.foreach_hyper_gradient(
            True, compute_contribution)
        with open(
            os.path.join(args.save_dir, str(epoch) +
                         ".contribution.json"), "wt"
        ) as f:
            json.dump(contribution_dict, f)

    hyper_gradient_trainer.train(after_epoch_callbacks=[after_epoch_callback])
