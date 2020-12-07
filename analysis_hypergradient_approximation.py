import argparse
import json
import os

import torch
from cyy_naive_lib.algorithm.mapping_op import (change_mapping_keys,
                                                dict_value_by_order)
from cyy_naive_pytorch_lib.visualization import EpochWindow, Window
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str)
parser.add_argument("--max_epoch", type=int, default=None)
parser.add_argument(
    "--analysis_influence_function",
    action="store_true",
    default=False)
args = parser.parse_args()


training_loss = []
with open(os.path.join(args.save_dir, "training_loss.json"), mode="rt") as f:
    training_loss = json.load(f)


validation_loss = {}
with open(os.path.join(args.save_dir, "validation_loss.json"), mode="rt") as f:
    validation_loss = change_mapping_keys(json.load(f), int)

assert len(training_loss) == len(validation_loss)
loss_win = EpochWindow("Training & Validation loss")
for idx, loss in enumerate(training_loss):
    loss_win.plot_loss(idx + 1, loss, "Training loss")
for epoch, loss in validation_loss.items():
    loss_win.plot_loss(epoch, loss, "Validation loss")

validation_accuracy = {}
with open(os.path.join(args.save_dir, "validation_accuracy.json"), mode="rt") as f:
    validation_accuracy = change_mapping_keys(json.load(f), int)

acc_win = EpochWindow("Validation accuracy")
acc_win.set_opt("showlegend", False)
for epoch, accu in validation_accuracy.items():
    acc_win.plot_accuracy(epoch, accu)

save_dir = args.save_dir

if args.max_epoch is None:
    args.max_epoch = len(training_loss)


mean_error = dict()

for epoch in range(1, args.max_epoch + 1):
    if os.path.isfile(
        os.path.join(
            save_dir,
            "approximation_distance_" +
            str(epoch) +
            ".json")
    ):
        with open(
            os.path.join(
                save_dir,
                "approximation_distance_" +
                str(epoch) +
                ".json"),
            "r",
        ) as f:
            distances = json.load(f)
            distance_mean = torch.Tensor(list(distances.values())).mean()
            mean_error[epoch] = distance_mean


error_win = EpochWindow("Mean approximation error", y_label="Mean Error")
error_win.set_opt("showlegend", False)
error_win.set_opt("ytickmin", 0)
for k in sorted(mean_error.keys()):
    error_win.plot_scalar(k, mean_error[k], name="Mean error")

epoch_contributions = dict()

for epoch in range(1, args.max_epoch + 1):
    if os.path.isfile(
        os.path.join(
            save_dir,
            "hessian_hyper_gradient_contribution.epoch_" +
            str(epoch) + ".json",
        )
    ):
        with open(
            os.path.join(
                save_dir,
                "hessian_hyper_gradient_contribution.epoch_" +
                str(epoch) + ".json",
            ),
            "r",
        ) as f:
            epoch_contributions[epoch] = json.load(f)

epoch_approximation_contributions = dict()
for epoch in range(1, args.max_epoch + 1):
    if os.path.isfile(
        os.path.join(
            save_dir,
            "approximation_hyper_gradient_contribution.epoch_" +
            str(epoch) + ".json",
        )
    ):
        with open(
            os.path.join(
                save_dir,
                "approximation_hyper_gradient_contribution.epoch_"
                + str(epoch)
                + ".json",
            ),
            "r",
        ) as f:
            epoch_approximation_contributions[epoch] = json.load(f)

epoch_classic_influence_contributions_contributions = dict()
for epoch in range(1, args.max_epoch + 1):
    if os.path.isfile(
        os.path.join(
            save_dir,
            "classic_influence_function_contribution_" + str(epoch) + ".json",
        )
    ):
        with open(
            os.path.join(
                save_dir,
                "classic_influence_function_contribution_" +
                str(epoch) + ".json",
            ),
            "r",
        ) as f:
            epoch_classic_influence_contributions_contributions[epoch] = json.load(
                f)

assert epoch_classic_influence_contributions_contributions

sign_win = EpochWindow("Different contribution sign rate", y_label="Rate")
sign_win.set_opt("ytickmin", 0)
relative_error_win = EpochWindow(
    "Relative contribution error",
    y_label="Error")

for epoch in sorted(epoch_approximation_contributions.keys()):
    approximation_contribution = epoch_approximation_contributions[epoch]
    contribution = epoch_contributions[epoch]
    assert len(approximation_contribution) == len(contribution)
    total_len = len(contribution)
    values = list()
    normal_indices = set()
    for idx in approximation_contribution.keys():
        if approximation_contribution[idx] * contribution[idx] < 0:
            values.append(contribution[idx])
        else:
            normal_indices.add(idx)
    sign_win.plot_scalar(epoch, len(values) / total_len, name="HYDRA")

    abs_relative_error = []
    for idx in sorted(normal_indices):
        assert approximation_contribution[idx] * contribution[idx] > 0
        abs_relative_error.append(
            abs(
                (approximation_contribution[idx] - contribution[idx])
                / contribution[idx]
            )
        )
    relative_error_win.plot_scalar(
        epoch, torch.Tensor(abs_relative_error).mean(), name="HYDRA"
    )

for epoch in sorted(
        epoch_classic_influence_contributions_contributions.keys()):
    influence_function_contribution = (
        epoch_classic_influence_contributions_contributions[epoch]
    )
    contribution = epoch_contributions[epoch]
    assert len(influence_function_contribution) == len(contribution)
    total_len = len(contribution)
    values = list()
    normal_indices = set()
    for idx in approximation_contribution.keys():
        if influence_function_contribution[idx] * contribution[idx] < 0:
            values.append(contribution[idx])
        else:
            normal_indices.add(idx)
    sign_win.plot_scalar(epoch, len(values) / total_len, name="influence")
    abs_relative_error = []
    for idx in sorted(normal_indices):
        assert contribution[idx] * influence_function_contribution[idx] > 0
        abs_relative_error.append(
            abs(
                (influence_function_contribution[idx] - contribution[idx])
                / contribution[idx]
            )
        )
    relative_error_win.plot_scalar(
        epoch, torch.Tensor(abs_relative_error).mean(), name="influence"
    )

spearmanr_win = EpochWindow(
    "Spearman's rank correlation coefficient", y_label="Coefficient"
)
for epoch in sorted(epoch_approximation_contributions.keys()):
    contribution_approximation_vector = list(
        dict_value_by_order(epoch_approximation_contributions[epoch])
    )
    contribution_vector = list(dict_value_by_order(epoch_contributions[epoch]))

    correlation = stats.spearmanr(
        contribution_vector, contribution_approximation_vector
    ).correlation
    spearmanr_win.plot_scalar(epoch, correlation, name="HYDRA")

for epoch in sorted(
        epoch_classic_influence_contributions_contributions.keys()):
    influence_function_contribution_vector = list(
        dict_value_by_order(
            epoch_classic_influence_contributions_contributions[epoch])
    )
    contribution_vector = list(dict_value_by_order(epoch_contributions[epoch]))
    correlation = stats.spearmanr(
        contribution_vector, influence_function_contribution_vector
    ).correlation
    spearmanr_win.plot_scalar(epoch, correlation, name="influence")
Window.save_envs()
