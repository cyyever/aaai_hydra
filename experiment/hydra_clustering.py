#!/usr/bin/env python3

import argparse
import json
import os
import pickle

import numpy as np
import torch
from cyy_naive_lib.algorithm.mapping_op import change_mapping_keys, flatten_mapping
from cyy_naive_lib.log import log_info
from cyy_torch_algorithm.hydra.hydra_analyzer import HyDRAAnalyzer
from cyy_torch_toolbox.dataset import (
    DatasetUtil,
    get_dataset_label_names,
    sample_subset,
)
from cyy_torch_toolbox.visualization import Window
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.manifold import TSNE

from .config import get_config


def compute_contribution(config, training_indices, label):
    result_dir = os.path.join(config.hydra_dir, "clustering", "label", str(label))
    os.makedirs(
        result_dir,
        exist_ok=True,
    )

    contribution_dict = None
    if os.path.isfile(os.path.join(result_dir, "contribution_dict.json")):
        with open(os.path.join(result_dir, "contribution_dict.json")) as f:
            contribution_dict = json.load(f)
            contribution_dict = change_mapping_keys(contribution_dict, int, True)
            log_info("use cached dict for label %s", label)
    else:
        tester = config.create_inferencer()

        test_subset = dict()

        sample_indices = sum(sample_subset(tester.dataset, 0.1).values(), [])
        for index in sample_indices:
            test_subset[index] = {index}

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

        contribution_dict = analyzer.get_training_sample_contributions(
            test_subset, training_indices
        )

        with open(os.path.join(result_dir, "contribution_dict.json"), mode="w") as f:
            json.dump(contribution_dict, f)
        assert set(contribution_dict.keys()) == set(training_indices)

    contribution_matrix = flatten_mapping(contribution_dict)
    return (contribution_dict, contribution_matrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hydra_dir", type=str, required=True)
    config = get_config(parser)

    result_dir = os.path.join(config.hydra_dir, "hydra_clustering")
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
        with open(
            os.path.join(result_dir, "contribution_list_" + str(label) + ".txt"),
            mode="w",
        ) as f:
            for k, v in contribution_dict.items():
                spaced_row: str = " ".join([str(a) for a in flatten_mapping(v)])
                if k in randomized_label_map:
                    spaced_row = "fake " + spaced_row
                else:
                    spaced_row = "real " + spaced_row
                print(spaced_row, file=f)

        with open(
            os.path.join(
                result_dir, "distribution_of_contribution_list_" + str(label) + ".txt"
            ),
            mode="w",
        ) as f:
            fake_matrix = []
            real_matrix = []
            for k, v in contribution_dict.items():
                v = flatten_mapping(v)
                if k in randomized_label_map:
                    fake_matrix.append(v)
                else:
                    real_matrix.append(v)
            print("fake std is ", np.std(fake_matrix, axis=0), file=f)
            print("real std is ", np.std(real_matrix, axis=0), file=f)
            print("fake mean is ", np.mean(fake_matrix, axis=0), file=f)
            print("real mean is ", np.mean(real_matrix, axis=0), file=f)

        log_info("get contribution_matrix for label %s", label)
        contribution_array = np.array(contribution_matrix)

        contribution_array = (
            contribution_array - contribution_array.mean(axis=0)
        ) / contribution_array.std(axis=0)

        tsne_res = TSNE(n_components=3).fit_transform(contribution_array)
        is_real_label = torch.ones((len(indices), 1))
        for i, index in enumerate(indices):
            if index in randomized_label_map:
                is_real_label[i] = 2
            else:
                is_real_label[i] = 1

        dataset_name = config.dc_config.dataset_name
        title = (
            "clustering for "
            + dataset_name
            + " class "
            + get_dataset_label_names(dataset_name)[label]
        )
        win = Window(title=title, env=config.dc_config.dataset_name + "_clustering")
        win.set_opt("legend", ["correct", "fake"])
        win.set_opt("markersize", 2)
        win.plot_scatter(x=torch.from_numpy(tsne_res), y=is_real_label)
        Window.save_envs()
        if dataset_name == "MNIST" or dataset_name == "FashionMNIST":
            clustering_res = KMeans(n_clusters=2).fit(contribution_array)
        elif dataset_name == "CIFAR10":
            # clustering_res = KMeans(n_clusters=2).fit(contribution_array)
            clustering_res = AgglomerativeClustering(
                n_clusters=2, affinity="cosine", linkage="average"
            ).fit(contribution_array)
        else:
            raise RuntimeError("Unknown dataset_name " + dataset_name)

        # clustering_res = AgglomerativeClustering(
        #     n_clusters=2, linkage="average").fit(tsne_res)
        # clustering_res = KMeans(n_clusters=2).fit(contribution_array)
        # clustering_res = SpectralClustering(n_clusters=2,
        # random_state=0).fit( contribution_array)

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
            dc.get_label_names()[label],
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
