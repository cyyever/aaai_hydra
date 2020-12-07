#!/usr/bin/env python3

import copy
import datetime
import os

import torch
from cyy_naive_lib.algorithm.mapping_op import get_mapping_values_by_order
from cyy_naive_lib.log import get_logger, set_file_handler
from cyy_naive_pytorch_lib.algorithm.influence_function.args import \
    add_arguments_to_parser
from cyy_naive_pytorch_lib.algorithm.influence_function.hyper_gradient_analyzer import \
    HyperGradientAnalyzer
from cyy_naive_pytorch_lib.arg_parse import (get_inferencer_from_args,
                                             get_parsed_args,
                                             get_training_dataset)
from cyy_naive_pytorch_lib.dataset import save_sample, sub_dataset


def get_instance_statistics(validator, instance_dataset):
    tmp_validator = copy.deepcopy(validator)
    tmp_validator.set_dataset(instance_dataset)
    other_data = tmp_validator.validate(1, per_sample_prob=True)[2]
    return other_data["per_sample_prob"][0]


def save_training_image(save_dir, validator,
                        contribution, training_dataset, index):
    sample_dataset = sub_dataset(training_dataset, [index])
    max_prob_index, max_prob = get_instance_statistics(
        validator, sample_dataset)

    save_sample(
        sample_dataset,
        0,
        os.path.join(
            save_dir,
            "index_{}_contribution_{}_predicted_class_{}_prob_{}_real_class_{}.jpg".format(
                index,
                contribution[index],
                max_prob_index,
                max_prob,
                sample_dataset[0][1],
            ),
        ),
    )


def save_test_image(save_dir, validator, contribution, index):
    sample_dataset = sub_dataset(validator.dataset, [index])
    max_prob_index, max_prob = get_instance_statistics(
        validator, sample_dataset)

    save_sample(
        sample_dataset,
        0,
        os.path.join(
            save_dir,
            "index_{}_contribution_{}_predicted_class_{}_prob_{}_real_class_{}.jpg".format(
                index,
                contribution[index],
                max_prob_index,
                max_prob,
                sample_dataset[0][1],
            ),
        ),
    )


if __name__ == "__main__":

    parser = add_arguments_to_parser()
    parser.add_argument("--sample_index", type=int, default=None)
    parser.add_argument("--threshold", type=float)
    args = get_parsed_args(parser)

    file_name = "{date:%Y-%m-%d_%H:%M:%S}".format(date=datetime.datetime.now())
    if args.sample_index is None:
        file_name += ".log"
    else:
        file_name += ".sample_" + str(args.sample_index) + ".log"
    set_file_handler(
        os.path.join(
            "log",
            "hypergradient_analysis",
            args.dataset_name,
            args.model_name,
            file_name,
        )
    )

    task_name = args.task_name
    training_dataset = get_training_dataset(args)
    validator = get_inferencer_from_args(args)
    analyzer = HyperGradientAnalyzer(validator, args.hyper_gradient_dir)
    contribution = None
    if args.sample_index is None:
        contribution_dict = analyzer.get_contributions()
        contribution = torch.Tensor(
            list(get_mapping_values_by_order(contribution_dict))
        )
    else:
        test_subset = dict()
        for idx in range(len(validator.dataset)):
            test_subset[idx] = [idx]
        contribution_dict = analyzer.get_subset_contributions(
            training_subset_dict={args.sample_index: [args.sample_index]},
            test_subset_dict=test_subset,
        )
        contribution = torch.Tensor(
            list(get_mapping_values_by_order(
                contribution_dict[args.sample_index]))
        )

    std, mean = torch.std_mean(contribution)
    max_contribution = torch.max(contribution)
    min_contribution = torch.min(contribution)

    get_logger().info("std is %s", std)
    get_logger().info("mean is %s", mean)
    get_logger().info("max contribution is %s", max_contribution)
    get_logger().info("min contribution is %s", min_contribution)
    get_logger().info(
        "positive contributions is %s", contribution[contribution >= 0].shape
    )
    get_logger().info(
        "negative contributions is %s", contribution[contribution < 0].shape
    )

    analysis_result_dir = os.path.join(
        "hypergradient_analysis_result", task_name)
    if args.sample_index is not None:
        analysis_result_dir = os.path.join(
            analysis_result_dir, "sample_" + str(args.sample_index)
        )

    mask = contribution > (max_contribution * args.threshold)
    for idx in mask.nonzero().tolist():
        idx = idx[0]
        if args.sample_index is None:
            save_training_image(
                analysis_result_dir, validator, contribution, training_dataset, idx
            )
        else:
            save_test_image(analysis_result_dir, validator, contribution, idx)

    mask = contribution < (min_contribution * args.threshold)
    for idx in mask.nonzero().tolist():
        idx = idx[0]
        if args.sample_index is None:
            save_training_image(
                analysis_result_dir, validator, contribution, training_dataset, idx
            )
        else:
            save_test_image(analysis_result_dir, validator, contribution, idx)
