#!/usr/bin/env python3

import copy
import datetime
import json
import os
import sys

import matplotlib
import numpy
import seaborn
from cyy_naive_lib.log import get_logger, set_file_handler
from cyy_naive_pytorch_lib.algorithm.influence_function.args import \
    add_arguments_to_parser
from cyy_naive_pytorch_lib.algorithm.influence_function.hyper_gradient_analyzer import \
    HyperGradientAnalyzer
from cyy_naive_pytorch_lib.algorithm.normalization import normalize_for_heatmap
from cyy_naive_pytorch_lib.arg_parse import (get_inferencer_from_args,
                                             get_parsed_args,
                                             get_training_dataset)
from cyy_naive_pytorch_lib.dataset import (get_dataset_labels,
                                           split_dataset_by_class)

matplotlib.use("Agg")


def compute_distribution(
    dataset_name,
    tester,
    training_dataset,
    hyper_gradient_dir,
    indices=None,
    prefix="whole",
):
    training_set_size = len(training_dataset)
    tester = copy.deepcopy(tester)
    analyzer = HyperGradientAnalyzer(tester, hyper_gradient_dir)

    training_subset = dict()
    for label, label_dataset in split_dataset_by_class(
            training_dataset).items():
        get_logger().info("compute label %s", label)
        training_subset[label] = set(label_dataset["indices"])
        if indices is not None:
            training_subset[label] &= set(indices)

    test_subset = dict()
    for label, label_dataset in split_dataset_by_class(tester.dataset).items():
        get_logger().info("compute label %s", label)
        test_subset[label] = set(label_dataset["indices"])
    subset_contribution_dict = analyzer.get_subset_contributions(
        training_subset, test_subset, training_set_size
    )
    means = dict()
    for training_label, tmp in subset_contribution_dict.items():
        means[training_label] = dict()
        for test_label, v in tmp.items():
            means[training_label][test_label] = v / (
                len(training_subset[training_label]) *
                len(test_subset[test_label])
            )

    result_dir = os.path.join(
        "hypergradient_distribution",
        dataset_name,
    )
    os.makedirs(
        result_dir,
        exist_ok=True,
    )

    if prefix:
        prefix = os.path.basename(hyper_gradient_dir) + "_" + prefix
    else:
        prefix = os.path.basename(hyper_gradient_dir)
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
                    line += " " + \
                        "{:e}".format(means[subset_label][test_label])
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

    label_list = get_dataset_labels(args.dataset_name)
    if args.task_name == "MNIST":
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
    parser = add_arguments_to_parser()
    args = get_parsed_args(parser)

    set_file_handler(
        os.path.join(
            "log",
            "hypergradient_distribution",
            args.dataset_name,
            args.model_name,
            "{date:%Y-%m-%d_%H:%M:%S}.log".format(
                date=datetime.datetime.now()),
        )
    )
    training_dataset = get_training_dataset(args)
    tester = get_inferencer_from_args(args)
    if args.randomized_label_map_path is None:
        compute_distribution(
            args.dataset_name,
            tester,
            training_dataset,
            args.hyper_gradient_dir,
        )
        sys.exit(0)

    with open(args.randomized_label_map_path, "r") as f:
        randomized_label_map = dict()
        for k, v in json.load(f).items():
            randomized_label_map[int(k)] = int(v)
        get_logger().info(
            "%s fake samples", len(randomized_label_map) /
            len(training_dataset)
        )

        compute_distribution(
            args.dataset_name,
            tester,
            training_dataset,
            args.hyper_gradient_dir,
            indices=randomized_label_map.keys(),
            prefix="abnormal",
        )

        compute_distribution(
            args.dataset_name,
            tester,
            training_dataset,
            args.hyper_gradient_dir,
            indices=set(range(len(training_dataset)))
            - set(randomized_label_map.keys()),
            prefix="normal",
        )
