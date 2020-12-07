#!/usr/bin/env python3

import datetime
import json
import os

import numpy as np
import torch
# from cyy_naive_lib.list_op import dict_to_list, change_dict_key
from cyy_naive_lib.algorithm.mapping_op import (change_mapping_keys,
                                                flatten_mapping)
from cyy_naive_lib.log import get_logger, set_file_handler
from cyy_naive_pytorch_lib.algorithm.influence_function.hyper_gradient_analyzer import \
    HyperGradientAnalyzer
from cyy_naive_pytorch_lib.arg_parse import (create_inferencer_from_args,
                                             get_randomized_label_map,
                                             get_training_dataset)
from cyy_naive_pytorch_lib.dataset import (get_dataset_label_names,
                                           sample_subset,
                                           split_dataset_by_class)
from cyy_naive_pytorch_lib.visualization import Window
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.manifold import TSNE

# from tools.configuration import get_task_configuration, get_task_dataset_name


def compute_contribution(args, label):
    result_dir = os.path.join(
        "hypergradient_clustering", args.task_name, "label", str(label)
    )
    os.makedirs(
        result_dir,
        exist_ok=True,
    )

    contribution_dict = None
    if os.path.isfile(os.path.join(result_dir, "contribution_dict.json")):
        with open(os.path.join(result_dir, "contribution_dict.json"), mode="rt") as f:
            contribution_dict = json.load(f)
            contribution_dict = change_mapping_keys(
                contribution_dict, int, True)
            get_logger().info("use cached dict for label %s", label)
    else:
        training_dataset = get_training_dataset(args)
        training_subset_indices = split_dataset_by_class(training_dataset)[label][
            "indices"
        ]

        validator = create_inferencer_from_args(args)

        test_subset = dict()

        sample_indices = sum(
            sample_subset(
                validator.dataset,
                0.1).values(),
            [])
        for index in sample_indices:
            test_subset[index] = {index}

        analyzer = HyperGradientAnalyzer(
            validator, args.hyper_gradient_dir, cache_size=args.cache_size
        )

        contribution_dict = analyzer.get_training_sample_contributions(
            test_subset, training_subset_indices
        )

        with open(os.path.join(result_dir, "contribution_dict.json"), mode="wt") as f:
            json.dump(contribution_dict, f)
        assert set(contribution_dict.keys()) == set(training_subset_indices)

    if args.use_sign_feature:
        get_logger().info("use sign feature")
        for v in contribution_dict.values():
            for k2, v2 in v.items():
                if v2 >= 0:
                    v[k2] = 1
                else:
                    v[k2] = 0
    else:
        get_logger().info("not use sign feature")

    contribution_matrix = flatten_mapping(contribution_dict)
    return (contribution_dict, contribution_matrix)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str)
    parser.add_argument("--hyper_gradient_dir", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--cache_size", type=int, default=1024)
    parser.add_argument("--randomized_label_map_path", type=str)
    parser.add_argument("--check_label", type=int)
    parser.add_argument(
        "--use_sign_feature",
        action="store_true",
        default=False)
    args = parser.parse_args()

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

    training_dataset = get_training_dataset(args)

    positive_overrates = []
    negative_overrates = []
    for label, v in split_dataset_by_class(training_dataset).items():
        if args.check_label is not None and label != args.check_label:
            continue
        indices = v["indices"]
        (contribution_dict, contribution_matrix) = compute_contribution(args, label)
        with open(
            os.path.join(
                result_dir,
                "contribution_list_" +
                str(label) +
                ".txt"),
            mode="wt",
        ) as f:
            for k, v in contribution_dict.items():
                spaced_row: str = " ".join([str(a)
                                            for a in flatten_mapping(v)])
                if k in randomized_label_map:
                    spaced_row = "fake " + spaced_row
                else:
                    spaced_row = "real " + spaced_row
                print(spaced_row, file=f)

        with open(
            os.path.join(
                result_dir, "distribution_of_contribution_list_" +
                str(label) + ".txt"
            ),
            mode="wt",
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

        get_logger().info("get contribution_matrix for label %s", label)
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

        dataset_name = args.dataset_name
        title = (
            "clustering for "
            + dataset_name
            + " class "
            + get_dataset_label_names(dataset_name)[label]
        )
        if args.use_sign_feature:
            title += "_and_sign_feature"
        win = Window(title=title, env=args.task_name + "_clustering")
        win.set_opt("legend", ["correct", "fake"])
        win.set_opt("markersize", 2)
        win.plot_scatter(x=torch.from_numpy(tsne_res), y=is_real_label)
        Window.save_envs()
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

        # clustering_res = AgglomerativeClustering(
        #     n_clusters=2, linkage="average").fit(tsne_res)
        # clustering_res = KMeans(n_clusters=2).fit(contribution_array)
        # clustering_res = SpectralClustering(n_clusters=2,
        # random_state=0).fit( contribution_array)

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
