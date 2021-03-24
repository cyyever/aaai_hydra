#!/usr/bin/env python3
import datetime
import json
import os

from cyy_naive_lib.algorithm.mapping_op import get_mapping_values_by_order
from cyy_naive_lib.log import get_logger, set_file_handler
from cyy_naive_pytorch_lib.arg_parse import (create_trainer_from_args,
                                             get_arg_parser, get_parsed_args)
from cyy_naive_pytorch_lib.dataset import sub_dataset

if __name__ == "__main__":
    parser = get_arg_parser()
    parser.add_argument("--contribution_dir", type=str)
    args = get_parsed_args(parser)

    set_file_handler(
        os.path.join(
            "log",
            "train_without_bad_samples",
            args.dataset_name,
            args.model_name,
            "{date:%Y-%m-%d_%H:%M:%S}.log".format(
                date=datetime.datetime.now()),
        )
    )

    trainer = create_trainer_from_args(args)
    full_training_dataset = trainer.dataset

    # for epoch in range(1, args.epochs + 1):
    for epoch in range(args.epochs, args.epochs + 1):
        contribution_path = os.path.join(
            args.contribution_dir, str(epoch) + ".contribution.json"
        )
        print(contribution_path)
        if not os.path.exists(contribution_path):
            break
        print("process epoch", epoch)
        print("drop negative indices")
        contribution_dict = None
        with open(contribution_path, "r") as f:
            contribution_dict = json.load(f)

        contribution = sorted(
            list(get_mapping_values_by_order(contribution_dict)))
        contribution = contribution[int(len(contribution) * 0.8):]
        print("contribution number is", len(contribution))
        positive_indices = []
        boundary_contribution = contribution[0]
        for k, v in contribution_dict.items():
            if v >= boundary_contribution:
                positive_indices.append(int(k))
        trainer.set_training_dataset(
            sub_dataset(full_training_dataset, positive_indices)
        )

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
