#!/usr/bin/env python3
import copy
import datetime
import json
import os

from cyy_naive_lib.log import get_logger, set_file_handler
from cyy_torch_algorithm.influence_function.classic_influence_function import \
    compute_classic_influence_function
from cyy_naive_pytorch_lib.arg_parse import (create_inferencer_from_args,
                                             create_trainer_from_args,
                                             get_parsed_args,
                                             get_training_dataset)
from cyy_naive_pytorch_lib.dataset import (sample_subset, split_dataset,
                                           sub_dataset)
from cyy_naive_pytorch_lib.gradient import get_dataset_gradients


def compute_contribution(args):
    result_dir = os.path.join(
        "hypergradient_clustering_for_influence_function",
        args.task_name,
    )
    os.makedirs(
        result_dir,
        exist_ok=True,
    )

    trainer = create_trainer_from_args(args)
    trainer.set_training_dataset(get_training_dataset(args))
    tester = create_inferencer_from_args(args)
    trainer.load_model(args.model_path)
    tester.load_model(args.model_path)

    training_sub_datasets = dict()
    for index, training_sub_dataset in enumerate(
        split_dataset(trainer.dataset)
    ):
        training_sub_datasets[index] = training_sub_dataset

    get_logger().info("begin get_dataset_gradients")
    training_sample_gradients = get_dataset_gradients(
        training_sub_datasets, tester)
    get_logger().info("end get_dataset_gradients")

    test_sample_indices = sum(sample_subset(tester.dataset, 0.1).values(), [])

    get_logger().info("test_sample_indices len is %s", len(test_sample_indices))
    contribution_dict = dict()
    for test_sample_index in test_sample_indices:
        test_sample_validator = copy.deepcopy(tester)
        test_sample_validator.set_dataset(
            sub_dataset(tester.dataset, [test_sample_index])
        )
        get_logger().info("compute test_sample_index %s", test_sample_index)
        contributions_from_influcence_function = compute_classic_influence_function(
            trainer,
            test_sample_validator.get_gradient(),
            training_sample_gradients,
            batch_size=args.batch_size,
            dampling_term=0.1,
            scale=10000,
            epsilon=0.03,
        )
        for (
            training_sample_index,
            contribution,
        ) in contributions_from_influcence_function.items():
            if training_sample_index not in contribution_dict:
                contribution_dict[int(training_sample_index)] = dict()
            contribution_dict[int(training_sample_index)][
                test_sample_index
            ] = contribution

    assert set(contribution_dict.keys()) == set(
        range(len(trainer.dataset)))
    with open(os.path.join(result_dir, "contribution_dict.json"), mode="wt") as f:
        json.dump(contribution_dict, f)


if __name__ == "__main__":
    args = get_parsed_args()

    set_file_handler(
        os.path.join(
            "log",
            "compute_classic_influence_function_for_clustering",
            args.dataset_name,
            args.model_name,
            "{date:%Y-%m-%d_%H:%M:%S}.log".format(
                date=datetime.datetime.now()),
        )
    )

    compute_contribution(args)
