import os

import cyy_torch_vision  # noqa: F401
import torch
from cyy_naive_lib.algorithm.mapping_op import get_mapping_values_by_key_order
from cyy_naive_lib.log import log_info
from cyy_torch_toolbox import (
    MachineLearningPhase,
    Trainer,
)
from cyy_torch_toolbox.metrics.prob_metric import ProbabilityMetric
from cyy_torch_vision import VisionDatasetUtil
from cyy_torch_xai import SampleContributions


def analysis_contribution(
    contribution_dict: SampleContributions, threshold: float
) -> tuple[dict, dict]:
    contribution = torch.Tensor(
        list(get_mapping_values_by_key_order(contribution_dict))
    )
    std, mean = torch.std_mean(contribution)
    max_contribution = torch.max(contribution).item()
    min_contribution = torch.min(contribution).item()

    log_info("std is %s", std)
    log_info("mean is %s", mean)
    log_info("max contribution is %s", max_contribution)
    log_info("min contribution is %s", min_contribution)
    log_info("positive contributions have %s", contribution[contribution >= 0].shape)
    log_info("negative contributions have %s", contribution[contribution < 0].shape)

    positive_contributions: dict = {
        k: v for k, v in contribution_dict.items() if v > (max_contribution * threshold)
    }
    negative_contributions: dict = {
        k: v for k, v in contribution_dict.items() if v < (min_contribution * threshold)
    }
    return positive_contributions, negative_contributions


def save_image(
    save_dir: str, executor: Trainer, contribution: dict, index: int
) -> None:
    label_names = executor.dataset_util.get_label_names()
    tester = executor.get_inferencer(
        phase=MachineLearningPhase.Training, copy_dataset=True
    )
    tester.dataset_collection.set_subset(
        phase=MachineLearningPhase.Training, indices={index}
    )
    prob_metric = ProbabilityMetric()
    tester.append_hook(prob_metric, "prob")
    tester.get_sample_loss()
    prob_index, prob = prob_metric.get_prob(epoch=1)[index]

    util = tester.dataset_util
    assert isinstance(util, VisionDatasetUtil)
    util.save_sample_image(
        0,
        path=os.path.join(
            save_dir,
            f"index_{index}_contribution_{contribution[index]}_predicted_class_{label_names[prob_index]}_prob_{prob}_real_class_{label_names[list(tester.dataset_util.get_labels())[0]]}.jpg",
        ),
    )
