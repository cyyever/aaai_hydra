#!/usr/bin/env python3

import argparse
import datetime
import json
import os
import pickle
import sys

import matplotlib
import numpy
import seaborn
from cyy_naive_lib.algorithm.mapping_op import get_mapping_values_by_key_order
from cyy_naive_lib.log import log_info, set_file_handler
from cyy_torch_algorithm.hydra.hydra_analyzer import HyDRAAnalyzer
from cyy_torch_algorithm.hydra.hydra_config import HyDRAConfig
from cyy_torch_algorithm.normalization import normalize_for_heatmap
from cyy_torch_toolbox.ml_type import MachineLearningPhase
from cyy_torch_toolbox.trainer import Trainer

matplotlib.use("Agg")


def compute_distribution(
    config,
    trainer: Trainer,
    indices=None,
    prefix="whole",
):
    training_set_size = None
    with open(
        os.path.join(config.hydra_dir, "training_set_size"),
        mode="rb",
    ) as f:
        training_set_size = pickle.load(f)
    analyzer = HyDRAAnalyzer(
        trainer.get_inferencer(),
        os.path.join(config.hydra_dir, "approximation_hyper_gradient_dir"),
        training_set_size,
    )

    if prefix == "abnormal":
        indices = set(range(training_set_size)) - indices

    training_subset = {}
    dataset_util = trainer.dataset_collection.get_dataset_util(
        MachineLearningPhase.Training
    )
    for label, label_indices in dataset_util.label_sample_dict.items():
        log_info("compute label %s", label)
        training_subset[label] = set(label_indices)
        if indices is not None:
            training_subset[label] &= set(indices)

    test_subset = {}
    dataset_util = trainer.dataset_collection.get_dataset_util(
        MachineLearningPhase.Test
    )
    for label, label_dataset in dataset_util.label_sample_dict.items():
        log_info("compute label %s", label)
        test_subset[label] = set(label_dataset)

    subset_contribution_dict = analyzer.get_subset_contributions(
        training_subset, test_subset
    )
    means = {}
    for training_label, tmp in subset_contribution_dict.items():
        means[training_label] = {}
        for test_label, v in tmp.items():
            means[training_label][test_label] = v / (
                len(training_subset[training_label]) * len(test_subset[test_label])
            )

    result_dir = os.path.join(
        "hydra_distribution",
        config.dataset_name,
    )
    os.makedirs(
        result_dir,
        exist_ok=True,
    )

    if prefix:
        prefix = os.path.basename(config.hydra_dir) + "_" + prefix
    else:
        prefix = os.path.basename(config.hydra_dir)
    with open(
        os.path.join(
            result_dir,
            prefix + ".mean.json",
        ),
        mode="w",
    ) as f:
        json.dump(means, f)

    means_list = []
    diagonal_sum = 0
    with open(
        os.path.join(
            result_dir,
            prefix + ".mean.txt",
        ),
        mode="w",
    ) as f:
        for subset_label in sorted(means.keys()):
            line = ""
            sub_list = []
            for test_label in sorted(means[subset_label].keys()):
                sub_list.append(means[subset_label][test_label])
                if subset_label == test_label:
                    line += (
                        " \\mathbf{\\textcolor{red}{"
                        + f"{means[subset_label][test_label]:e}"
                        + "}}"
                    )
                    diagonal_sum += means[subset_label][test_label]
                else:
                    line += " " + f"{means[subset_label][test_label]:e}"
            print(line, file=f)
            means_list.append(sub_list)

    with open(
        os.path.join(
            result_dir,
            prefix + ".diagonal_sum.txt",
        ),
        mode="w",
    ) as f:
        print(diagonal_sum, file=f)

    mean_array = numpy.array(means_list)
    mean_array = mean_array - mean_array.mean()
    mean_array = mean_array / mean_array.std()

    label_list = get_mapping_values_by_key_order(dataset_util.get_label_names())
    if config.dataset_name == "MNIST":
        label_list = [" " * 18 + str(a) for a in label_list]
    mean_array = normalize_for_heatmap(mean_array)

    seaborn.set(font_scale=2)

    ax = seaborn.heatmap(
        mean_array,
        xticklabels=label_list,
        yticklabels=label_list,
        cbar=True,
    )
    ax.get_figure().savefig(
        os.path.join(result_dir, prefix + ".mean_heat.jpg"), bbox_inches="tight"
    )
    ax.get_figure().clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hydra_dir", type=str, required=True)
    config = HyDRAConfig()

    set_file_handler(
        os.path.join(
            "log",
            "hydra_distribution",
            config.dc_config.dataset_name,
            config.model_config.model_name,
            f"{datetime.datetime.now():%Y-%m-%d_%H:%M:%S}.log",
        )
    )
    trainer = config.create_trainer()
    randomized_label_map = config.dc_config.training_dataset_label_map
    if randomized_label_map is None:
        compute_distribution(config, trainer)
        sys.exit(0)

    compute_distribution(
        config, trainer, indices=randomized_label_map.keys(), prefix="abnormal"
    )

    compute_distribution(
        config,
        trainer,
        indices=randomized_label_map.keys(),
        prefix="normal",
    )
