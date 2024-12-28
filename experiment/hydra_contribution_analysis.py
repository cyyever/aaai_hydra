#!/usr/bin/env python3

import argparse
import copy
import json
import os
import pickle

import torch
from config import get_config
from cyy_naive_lib.algorithm.mapping_op import (
    change_mapping_keys,
    get_mapping_values_by_key_order,
)
from cyy_naive_lib.log import log_info
from cyy_torch_algorithm.hydra.hydra_analyzer import HyDRAAnalyzer
from cyy_torch_toolbox import DatasetUtil, MachineLearningPhase, sub_dataset


def get_instance_statistics(tester, instance_dataset) -> dict:
    tmp_validator = copy.deepcopy(tester)
    tmp_validator.dataset_collection.transform_dataset(
        MachineLearningPhase.Test, lambda _: instance_dataset
    )
    tmp_validator.inference(sample_prob=True)
    return tmp_validator.prob_metric.get_prob(1)[0]


def save_training_image(save_dir, tester, contribution, training_dataset, index):
    sample_dataset = sub_dataset(training_dataset, [index])
    max_prob_index, max_prob = get_instance_statistics(tester, sample_dataset)

    DatasetUtil(sample_dataset).save_sample_image(
        0,
        os.path.join(
            save_dir,
            f"index_{index}_contribution_{contribution[index]}_predicted_class_{max_prob_index}_prob_{max_prob}_real_class_{sample_dataset[0][1]}.jpg",
        ),
    )


def save_test_image(save_dir, tester, contribution, index):
    sample_dataset = sub_dataset(tester.dataset, [index])
    max_prob_index, max_prob = get_instance_statistics(tester, sample_dataset)

    DatasetUtil(sample_dataset).save_sample_image(
        0,
        os.path.join(
            save_dir,
            f"index_{index}_contribution_{contribution[index]}_predicted_class_{max_prob_index}_prob_{max_prob}_real_class_{sample_dataset[0][1]}.jpg",
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

    contribution = None
    if args.sample_index is None:
        with open(
            os.path.join(args.hydra_dir, "approx_hydra_contribution.json"),
        ) as f:
            contribution_dict = json.load(f)
            contribution = torch.Tensor(
                list(get_mapping_values_by_key_order(contribution_dict))
            )
    else:
        tester = config.create_inferencer(phase=MachineLearningPhase.Test)
        test_subset = dict()
        for idx in range(len(tester.dataset)):
            test_subset[idx] = [idx]
        with open(
            os.path.join(args.hydra_dir, "training_set_size"),
            mode="rb",
        ) as f:
            training_set_size = pickle.load(f)
        analyzer = HyDRAAnalyzer(
            tester,
            os.path.join(args.hydra_dir, "approximation_hyper_gradient_dir"),
            training_set_size,
        )
        contribution_dict = analyzer.get_subset_contributions(
            training_subset_dict={args.sample_index: [args.sample_index]},
            test_subset_dict=test_subset,
        )

        contribution_dict = change_mapping_keys(contribution_dict, int, True)

        contribution = torch.Tensor(
            list(get_mapping_values_by_key_order(contribution_dict[args.sample_index]))
        )

    std, mean = torch.std_mean(contribution)
    max_contribution = torch.max(contribution)
    min_contribution = torch.min(contribution)

    log_info("std is %s", std)
    log_info("mean is %s", mean)
    log_info("max contribution is %s", max_contribution)
    log_info("min contribution is %s", min_contribution)
    log_info("positive contributions is %s", contribution[contribution >= 0].shape)
    log_info("negative contributions is %s", contribution[contribution < 0].shape)

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
                analysis_result_dir, tester, contribution, training_dataset, idx
            )
        else:
            save_test_image(analysis_result_dir, tester, contribution, idx)

    mask = contribution < (min_contribution * args.threshold)
    for idx in mask.nonzero().tolist():
        idx = idx[0]
        if args.sample_index is None:
            save_training_image(
                analysis_result_dir, tester, contribution, training_dataset, idx
            )
        else:
            save_test_image(analysis_result_dir, tester, contribution, idx)
