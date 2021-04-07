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
from cyy_naive_lib.log import get_logger, set_file_handler
from cyy_naive_pytorch_lib.algorithm.hydra.hydra_analyzer import HyDRAAnalyzer
from cyy_naive_pytorch_lib.algorithm.normalization import normalize_for_heatmap
from cyy_naive_pytorch_lib.dataset import DatasetUtil
from cyy_naive_pytorch_lib.ml_type import MachineLearningPhase

from .config import get_config

matplotlib.use("Agg")


def compute_distribution(
    config,
    tester,
    indices=None,
    prefix="whole",
):
    training_dataset = tester.dataset_collection.get_dataset(
        MachineLearningPhase.Training
    )
    training_set_size = None
    with open(
        os.path.join(config.hydra_dir, "training_set_size"),
        mode="rb",
    ) as f:
        training_set_size = pickle.load(f)
    analyzer = HyDRAAnalyzer(
        tester,
        os.path.join(config.hydra_dir, "approximation_hyper_gradient_dir"),
        training_set_size,
    )

    if prefix == "abnormal":
        indices = set(range(len(training_dataset))) - indices

    training_subset = dict()
    for label, label_dataset in DatasetUtil(training_dataset).split_by_label().items():
        get_logger().info("compute label %s", label)
        training_subset[label] = set(label_dataset["indices"])
        if indices is not None:
            training_subset[label] &= set(indices)

    test_subset = dict()
    for label, label_dataset in DatasetUtil(tester.dataset).split_by_label().items():
        get_logger().info("compute label %s", label)
        test_subset[label] = set(label_dataset["indices"])

    subset_contribution_dict = analyzer.get_subset_contributions(
        training_subset, test_subset
    )
    means = dict()
    for training_label, tmp in subset_contribution_dict.items():
        means[training_label] = dict()
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
        mode="wt",
    ) as f:
        json.dump(means, f)

    means_list = []
    diagonal_sum = 0
    with open(
        os.path.join(
            result_dir,
            prefix + ".mean.txt",
        ),
        mode="wt",
    ) as f:
        for subset_label in sorted(means.keys()):
            line = ""
            sub_list = []
            for test_label in sorted(means[subset_label].keys()):
                sub_list.append(means[subset_label][test_label])
                if subset_label == test_label:
                    line += (
                        " \\mathbf{\\textcolor{red}{"
                        + "{:e}".format(means[subset_label][test_label])
                        + "}}"
                    )
                    diagonal_sum += means[subset_label][test_label]
                else:
                    line += " " + "{:e}".format(means[subset_label][test_label])
            print(line, file=f)
            means_list.append(sub_list)

    with open(
        os.path.join(
            result_dir,
            prefix + ".diagonal_sum.txt",
        ),
        mode="wt",
    ) as f:
        print(diagonal_sum, file=f)

    mean_array = numpy.array(means_list)
    mean_array = mean_array - mean_array.mean()
    mean_array = mean_array / mean_array.std()

    label_list = tester.dataset_collection.get_label_names()
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
    config = get_config(parser)

    set_file_handler(
        os.path.join(
            "log",
            "hydra_distribution",
            config.dc_config.dataset_name,
            config.model_name,
            "{date:%Y-%m-%d_%H:%M:%S}.log".format(date=datetime.datetime.now()),
        )
    )
    tester = config.create_inferencer()
    randomized_label_map = config.dc_config.training_dataset_label_map
    if randomized_label_map is None:
        compute_distribution(config, tester)
        sys.exit(0)

    compute_distribution(
        config, tester, indices=randomized_label_map.keys(), prefix="abnormal"
    )

    compute_distribution(
        config,
        tester,
        indices=randomized_label_map.keys(),
        prefix="normal",
    )
