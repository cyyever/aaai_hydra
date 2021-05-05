import argparse
import copy
import json
import os

import torch
from cyy_naive_lib.algorithm.mapping_op import (
    change_mapping_keys, get_mapping_values_by_key_order)
from cyy_naive_lib.log import get_logger
from cyy_naive_pytorch_lib.dataset import DatasetUtil
from cyy_naive_pytorch_lib.dataset_collection import DatasetCollection
from cyy_naive_pytorch_lib.ml_type import MachineLearningPhase


def get_instance_statistics(tester, instance_dataset) -> dict:
    tmp_validator = copy.deepcopy(tester)
    tmp_validator.dataset_collection.transform_dataset(
        MachineLearningPhase.Test, lambda _: instance_dataset
    )
    tmp_validator.inference(sample_prob=True)
    return tmp_validator.prob_metric.get_prob(1)[0]


def save_training_image(save_dir, training_dataset, index):
    image_path = os.path.join(save_dir, "index_{}.jpg".format(index))

    DatasetUtil(training_dataset).save_sample_image(index, image_path)
    return image_path


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True)
args = parser.parse_args()

dc = DatasetCollection.get_by_name(args.dataset_name)
json_path = os.path.join(
    "contribution", args.dataset_name, "approx_hydra_contribution.json"
)
if not os.path.isfile(json_path):
    raise RuntimeError("unknown dataset:" + args.dataset_name)

with open(json_path, "rt") as f:
    contribution_dict = json.load(f)
contribution_dict = change_mapping_keys(contribution_dict, int, True)
keys = list(sorted(contribution_dict.keys()))
contribution = torch.Tensor(list(get_mapping_values_by_key_order(contribution_dict)))

std, mean = torch.std_mean(contribution)
max_contribution = torch.max(contribution)
min_contribution = torch.min(contribution)

get_logger().info("std is %s", std)
get_logger().info("mean is %s", mean)
get_logger().info("max contribution is %s", max_contribution)
get_logger().info("min contribution is %s", min_contribution)
get_logger().info("positive contributions is %s", contribution[contribution >= 0].shape)
get_logger().info("negative contributions is %s", contribution[contribution < 0].shape)

image_info = dict()
mask = contribution >= max_contribution * 0.9
for idx in mask.nonzero().tolist():
    idx = keys[idx[0]]
    image_dir = os.path.join("image", args.dataset_name, "positive")
    os.makedirs(image_dir, exist_ok=True)
    training_dataset = dc.get_training_dataset()
    image_path = save_training_image(image_dir, training_dataset, idx)
    label = training_dataset[idx][1]
    image_info[str(idx)] = [
        image_path,
        dc.get_label_names()[label],
        contribution_dict[idx],
    ]


mask = contribution < min_contribution * 0.9
for idx in mask.nonzero().tolist():
    idx = keys[idx[0]]
    training_dataset = dc.get_training_dataset()
    image_dir = os.path.join("image", args.dataset_name, "negative")
    os.makedirs(image_dir, exist_ok=True)
    training_dataset = dc.get_training_dataset()
    image_path = save_training_image(image_dir, training_dataset, idx)
    label = training_dataset[idx][1]
    image_info[str(idx)] = [
        image_path,
        dc.get_label_names()[label],
        contribution_dict[idx],
    ]
with open(args.dataset_name + ".result.json", "wt") as f:
    json.dump(image_info, f)
