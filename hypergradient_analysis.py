#!/usr/bin/env python3

import argparse
import copy
import json
import os
import pickle

import torch
from cyy_naive_lib.algorithm.mapping_op import get_mapping_values_by_order
from cyy_naive_lib.log import get_logger
from cyy_naive_pytorch_lib.algorithm.hydra.hydra_analyzer import HyDRAAnalyzer
from cyy_naive_pytorch_lib.dataset import DatasetUtil, sub_dataset
from cyy_naive_pytorch_lib.ml_type import MachineLearningPhase

from config import get_config


def get_instance_statistics(validator, instance_dataset) -> dict:
    tmp_validator = copy.deepcopy(validator)
    tmp_validator.dataset_collection.transform_dataset(
        MachineLearningPhase.Test, lambda _: instance_dataset
    )
    tmp_validator.inference(sample_prob=True)
    return tmp_validator.prob_metric.get_prob(1)[0]


def save_training_image(save_dir, validator, contribution, training_dataset, index):
    sample_dataset = sub_dataset(training_dataset, [index])
    max_prob_index, max_prob = get_instance_statistics(validator, sample_dataset)

    DatasetUtil(sample_dataset).save_sample_image(
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
    max_prob_index, max_prob = get_instance_statistics(validator, sample_dataset)

    DatasetUtil(sample_dataset).save_sample_image(
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_index", type=int, default=None)
    parser.add_argument("--hydra_dir", type=str, required=True)
    parser.add_argument("--threshold", type=float)
    config = get_config(parser=parser)
    args = parser.parse_args()
    trainer = config.create_trainer()
    training_dataset = trainer.dataset
    validator = config.create_inferencer(phase=MachineLearningPhase.Test)

    contribution = None
    if args.sample_index is None:
        with open(
            os.path.join(args.hydra_dir, "approx_hydra_contribution.json"),
            mode="rt",
        ) as f:
            contribution_dict = json.load(f)
            contribution = torch.Tensor(
                list(get_mapping_values_by_order(contribution_dict))
            )
    else:
        test_subset = dict()
        for idx in range(len(validator.dataset)):
            test_subset[idx] = [idx]
        training_set_size = None
        with open(
            os.path.join(args.hydra_dir, "training_set_size"),
            mode="rb",
        ) as f:
            training_set_size = pickle.load(f)
        analyzer = HyDRAAnalyzer(
            validator,
            os.path.join(args.hydra_dir, "approximation_hyper_gradient_dir"),
            training_set_size,
        )
        contribution_dict = analyzer.get_subset_contributions(
            training_subset_dict={args.sample_index: [args.sample_index]},
            test_subset_dict=test_subset,
        )
        contribution = torch.Tensor(
            list(get_mapping_values_by_order(contribution_dict[args.sample_index]))
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

    analysis_result_dir = os.path.join(args.hydra_dir, "hydra_analysis_result")
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
