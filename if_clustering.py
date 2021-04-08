#!/usr/bin/env python3

import datetime
import json
import os

import numpy as np
from cyy_naive_lib.algorithm.mapping_op import (change_mapping_keys,
                                                flatten_mapping)
from cyy_naive_lib.log import get_logger, set_file_handler
from cyy_naive_pytorch_lib.algorithm.influence_function.args import \
    add_arguments_to_parser
from cyy_naive_pytorch_lib.arg_parse import (get_parsed_args,
                                             get_randomized_label_map,
                                             get_training_dataset)
from cyy_naive_pytorch_lib.dataset import (get_dataset_label_names,
                                           split_dataset_by_class)
from sklearn.cluster import AgglomerativeClustering, KMeans


def compute_contribution(args, label):
    result_dir = os.path.join(
        "hypergradient_clustering", args.task_name, "label", str(label)
    )
    os.makedirs(
        result_dir,
        exist_ok=True,
    )

    training_dataset = get_training_dataset(args)
    training_subset_indices = set(
        split_dataset_by_class(training_dataset)[label]["indices"]
    )
    contribution_dict = dict()
    with open(args.contribution_dict_path, mode="rt") as f:
        all_contribution_dict = change_mapping_keys(json.load(f), int, True)
        for index, v in all_contribution_dict.items():
            if index in training_subset_indices:
                contribution_dict[index] = v

    if args.use_sign_feature:
        get_logger().info("use sign feature")
        for v in contribution_dict.values():
            for k2, v2 in v.items():
                if v2 >= 0:
                    v[k2] = 1
                else:
                    v[k2] = 0
    assert set(contribution_dict.keys()) == set(training_subset_indices)

    contribution_matrix = flatten_mapping(contribution_dict)
    return (contribution_dict, contribution_matrix)


if __name__ == "__main__":
    parser = add_arguments_to_parser()
    parser.add_argument("--contribution_dict_path", type=str)
    parser.add_argument(
        "--use_sign_feature",
        action="store_true",
        default=False)
    args = get_parsed_args(parser)

    set_file_handler(
        os.path.join(
            "log",
            "hypergradient_clustering",
            args.task_name,
            "{date:%Y-%m-%d_%H:%M:%S}.log".format(
                date=datetime.datetime.now()),
        )
    )
    result_dir = os.path.join(
        "hypergradient_clustering",
        args.task_name,
    )
    os.makedirs(
        result_dir,
        exist_ok=True,
    )

    randomized_label_map = get_randomized_label_map(args)
    noisy_label_dict: dict = dict()
    for k, v in randomized_label_map.items():
        if v not in noisy_label_dict:
            noisy_label_dict[v] = set()
        noisy_label_dict[v].add(int(k))

    dataset_name = args.dataset_name
    training_dataset = get_training_dataset(args)

    positive_overrates = []
    negative_overrates = []
    for label, v in split_dataset_by_class(training_dataset).items():
        if args.check_label is not None and label != args.check_label:
            continue
        indices = v["indices"]
        (contribution_dict, contribution_matrix) = compute_contribution(args, label)

        get_logger().info("get contribution_matrix for label %s", label)
        contribution_array = np.array(contribution_matrix)
        if dataset_name == "MNIST":
            clustering_res = KMeans(n_clusters=2).fit(contribution_array)
        elif dataset_name == "FashionMNIST":
            clustering_res = KMeans(n_clusters=2).fit(contribution_array)
        elif dataset_name == "CIFAR10":
            # clustering_res = KMeans(n_clusters=2).fit(contribution_array)
            clustering_res = AgglomerativeClustering(
                n_clusters=2, affinity="cosine", linkage="average"
            ).fit(contribution_array)
        else:
            raise RuntimeError("Unknown dataset_name " + dataset_name)

        assert len(indices) == len(clustering_res.labels_)
        clusters: list = [set(), set()]
        for index, cluster_id in zip(indices, clustering_res.labels_):
            clusters[cluster_id].add(index)

        noisy_labels = set(noisy_label_dict[label])
        normal_labels = set(indices) - noisy_labels
        normal_cluster = clusters[0]
        noisy_cluster = clusters[1]
        if len(normal_labels & clusters[1]) > len(normal_labels & clusters[0]):
            normal_cluster = clusters[1]
            noisy_cluster = clusters[0]

        get_logger().info(
            "class %s,normal_cluster len is %s,noisy_cluster len is %s",
            get_dataset_label_names(dataset_name)[label],
            len(normal_cluster),
            len(noisy_cluster),
        )
        get_logger().info(
            "class %s,normal_labels len is %s,noisy_labels len is %s",
            get_dataset_label_names(dataset_name)[label],
            len(normal_labels),
            len(noisy_labels),
        )
        positive_overrates.append(
            len(normal_labels & normal_cluster) /
            len(normal_labels | normal_cluster)
        )
        negative_overrates.append(
            len(noisy_labels & noisy_cluster) /
            len(noisy_labels | noisy_cluster)
        )
        get_logger().info(
            "class %s,overlay rate in normal_cluster is %s,noisy_cluster is %s",
            get_dataset_label_names(dataset_name)[label],
            positive_overrates[-1],
            negative_overrates[-1],
        )

    get_logger().info(
        "average overlay rate in normal_cluster is %s,noisy_cluster is %s",
        np.mean(positive_overrates),
        np.mean(negative_overrates),
    )
