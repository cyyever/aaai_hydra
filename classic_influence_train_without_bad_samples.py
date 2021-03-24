#!/usr/bin/env python3
import datetime
import json
import os

from cyy_naive_lib.algorithm.mapping_op import get_mapping_values_by_order
from cyy_naive_lib.log import get_logger, set_file_handler
from cyy_naive_pytorch_lib.algorithm.influence_function.classic_influence_function import \
    compute_classic_influence_function
from cyy_naive_pytorch_lib.arg_parse import (create_inferencer_from_args,
                                             create_trainer_from_args,
                                             get_arg_parser, get_parsed_args)
from cyy_naive_pytorch_lib.dataset import sub_dataset
from cyy_naive_pytorch_lib.gradient import get_dataset_gradients

if __name__ == "__main__":
    parser = get_arg_parser()
    parser.add_argument("--model_dir", type=str)
    args = get_parsed_args(parser)

    set_file_handler(
        os.path.join(
            "log",
            "classic_influence_train_without_bad_samples",
            args.dataset_name,
            args.model_name,
            "{date:%Y-%m-%d_%H:%M:%S}.log".format(
                date=datetime.datetime.now()),
        )
    )

    trainer = create_trainer_from_args(args)
    inferencer = create_inferencer_from_args(args)
    model_file = "model.pt"
    assert os.path.isfile(os.path.join(args.model_dir, model_file))
    trainer.load_model(os.path.join(args.model_dir, model_file))
    inferencer.load_model(os.path.join(args.model_dir, model_file))

    training_sub_datasets = dict()
    for index in range(len(trainer.dataset)):
        training_sub_datasets[index] = sub_dataset(
            trainer.dataset, [index])
    training_sample_gradients = get_dataset_gradients(
        training_sub_datasets, inferencer)
    print("compute_classic_influence_function")

    contribution_dict = compute_classic_influence_function(
        trainer,
        inferencer.get_gradient(),
        training_sample_gradients,
        batch_size=args.batch_size,
        dampling_term=0.01,
        scale=1000,
        epsilon=0.03,
    )

    with open(model_file + ".classic_influence.json", "wt") as f:
        json.dump(contribution_dict, f)

    trainer = create_trainer_from_args(args)
    contribution = sorted(list(get_mapping_values_by_order(contribution_dict)))
    contribution = contribution[int(len(contribution) * 0.8):]
    boundary_contribution = contribution[0]
    positive_indices = []
    for k, v in contribution_dict.items():
        if v >= boundary_contribution:
            positive_indices.append(int(k))

    trainer.set_training_dataset(
        sub_dataset(trainer.dataset, positive_indices)
    )
    print("begin repeated_train")
    results = trainer.repeated_train(
        5, save_dir=None, plot_class_accuracy=False, test_epoch_interval=1
    )
    get_logger().info("training_loss is %s", results["training_loss"])
    get_logger().info("validation_loss is %s", results["validation_loss"])
    get_logger().info(
        "validation_accuracy is %s",
        results["validation_accuracy"])
    get_logger().info("test_loss is %s", results["test_loss"])
    get_logger().info("test_accuracy is %s", results["test_accuracy"])
