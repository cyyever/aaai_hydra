#!/usr/bin/env python3
import json
import os

from cyy_naive_pytorch_lib.algorithm.influence_function.classic_influence_function import \
    compute_classic_influence_function
from cyy_naive_pytorch_lib.arg_parse import (create_inferencer_from_args,
                                             create_trainer_from_args,
                                             get_arg_parser, get_parsed_args)
from cyy_naive_pytorch_lib.dataset import sub_dataset
from cyy_naive_pytorch_lib.gradient import get_dataset_gradients

# from tools.configuration import get_task_configuration

if __name__ == "__main__":
    parser = get_arg_parser()
    parser.add_argument("--min_epoch", type=int)
    parser.add_argument("--max_epoch", type=int, default=1000)

    args = get_parsed_args(parser)
    trainer = create_trainer_from_args(args)
    validator = create_inferencer_from_args(args)
    hyper_gradient_indices = []
    hyper_gradient_indices_path = os.path.join(
        args.save_dir, "..", "hyper_gradient_indices.json"
    )
    assert os.path.isfile(hyper_gradient_indices_path)
    if os.path.isfile(hyper_gradient_indices_path):
        with open(
            hyper_gradient_indices_path,
            mode="rt",
        ) as f:
            hyper_gradient_indices = json.load(f)
            print("use", len(hyper_gradient_indices), "indices")
    else:
        hyper_gradient_indices = list(range(len(trainer.training_dataset)))
    for epoch in range(args.min_epoch, args.max_epoch):
        model_path = os.path.join(
            args.save_dir,
            "model_epoch_" +
            str(epoch) +
            ".pt")
        if not os.path.isfile(model_path):
            continue
        if os.path.isfile(
            os.path.join(
                args.save_dir,
                "classic_influence_function_contribution_" +
                str(epoch) + ".json",
            )
        ):
            continue
        print("do epoch", epoch)
        trainer.load_model(model_path)
        validator.load_model(model_path)

        training_sub_datasets = dict()
        for hyper_gradient_index in hyper_gradient_indices:
            training_sub_datasets[hyper_gradient_index] = sub_dataset(
                trainer.training_dataset, [hyper_gradient_index]
            )

        training_sample_gradients = get_dataset_gradients(
            training_sub_datasets, validator
        )

        contributions = compute_classic_influence_function(
            trainer,
            validator.get_gradient(),
            training_sample_gradients,
            batch_size=args.batch_size,
            dampling_term=0.01,
            scale=1000,
            epsilon=0.03,
        )
        with open(
            os.path.join(
                args.save_dir,
                "classic_influence_function_contribution_" +
                str(epoch) + ".json",
            ),
            mode="wt",
        ) as f:
            json.dump(contributions, f)
