#!/usr/bin/env python3

import argparse
import json
import os

import numpy as np
from cyy_naive_lib.algorithm.mapping_op import change_mapping_keys, flatten_mapping
from cyy_naive_lib.log import log_info
from cyy_torch_toolbox.dataset import DatasetUtil, get_dataset_label_names
from sklearn.cluster import AgglomerativeClustering, KMeans

from .config import get_config


def compute_contribution(config, training_indices, label):
    result_dir = os.path.join(config.hydra_dir, "IF_clustering", "label", str(label))
    os.makedirs(
        result_dir,
        exist_ok=True,
    )

    contribution_dict = dict()
    with open(config.IF_dir) as f:
        all_contribution_dict = change_mapping_keys(json.load(f), int, True)
        for index, v in all_contribution_dict.items():
            if index in training_indices:
                contribution_dict[index] = v

    assert set(contribution_dict.keys()) == set(training_indices)

    contribution_matrix = flatten_mapping(contribution_dict)
    return (contribution_dict, contribution_matrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--IF_dir", type=str)
    config = get_config(parser)
    result_dir = os.path.join(config.IF_dir, "IF_clustering")
    os.makedirs(
        result_dir,
        exist_ok=True,
    )

    randomized_label_map = config.dc_config.training_dataset_label_map
    noisy_label_dict: dict = dict()
    for k, v in randomized_label_map.items():
        if v not in noisy_label_dict:
            noisy_label_dict[v] = set()
        noisy_label_dict[v].add(int(k))

    dc = config.dc_config.create_dataset_collection()

    positive_overrates = []
    negative_overrates = []
    for label, v in DatasetUtil(dc.get_training_dataset()).split_by_label().items():
        indices = v["indices"]
        (contribution_dict, contribution_matrix) = compute_contribution(
            config, indices, label
        )

        log_info("get contribution_matrix for label %s", label)
        contribution_array = np.array(contribution_matrix)
        dataset_name = config.dc_config.dataset_name
        if dataset_name == "MNIST" or dataset_name == "FashionMNIST":
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
        for index, cluster_id in zip(indices, clustering_res.labels_, strict=False):
            clusters[cluster_id].add(index)

        noisy_labels = set(noisy_label_dict[label])
        normal_labels = set(indices) - noisy_labels
        normal_cluster = clusters[0]
        noisy_cluster = clusters[1]
        if len(normal_labels & clusters[1]) > len(normal_labels & clusters[0]):
            normal_cluster = clusters[1]
            noisy_cluster = clusters[0]

        log_info(
            "class %s,normal_cluster len is %s,noisy_cluster len is %s",
            get_dataset_label_names(dataset_name)[label],
            len(normal_cluster),
            len(noisy_cluster),
        )
        log_info(
            "class %s,normal_labels len is %s,noisy_labels len is %s",
            get_dataset_label_names(dataset_name)[label],
            len(normal_labels),
            len(noisy_labels),
        )
        positive_overrates.append(
            len(normal_labels & normal_cluster) / len(normal_labels | normal_cluster)
        )
        negative_overrates.append(
            len(noisy_labels & noisy_cluster) / len(noisy_labels | noisy_cluster)
        )
        log_info(
            "class %s,overlay rate in normal_cluster is %s,noisy_cluster is %s",
            get_dataset_label_names(dataset_name)[label],
            positive_overrates[-1],
            negative_overrates[-1],
        )

    log_info(
        "average overlay rate in normal_cluster is %s,noisy_cluster is %s",
        np.mean(positive_overrates),
        np.mean(negative_overrates),
    )
