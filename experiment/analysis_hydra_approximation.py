import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
from cyy_naive_lib.algorithm.mapping_op import get_mapping_values_by_key_order
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str)
args = parser.parse_args()
save_dir = args.save_dir


mean_error = {}

epochs = []
for epoch in range(1000):
    file = os.path.join(save_dir, "approximation_distance_" + str(epoch) + ".json")
    if os.path.isfile(file):
        epochs.append(epoch)
        with open(file, encoding="utf8") as f:
            distances = json.load(f)
            distance_mean = torch.Tensor(list(distances.values())).mean()
            mean_error[epoch] = distance_mean

fig, ax = plt.subplots()
ax.set_title("Mean approximation error")
print(epochs)
print(list(get_mapping_values_by_key_order(mean_error)))
ax.plot(epochs, list(get_mapping_values_by_key_order(mean_error)), label="Mean error")
ax.legend()
fig.savefig("mean_error.jpg")

epoch_contributions = {}

for epoch in range(1000):
    file = os.path.join(
        save_dir, "hessian_hyper_gradient_contribution.epoch_" + str(epoch) + ".json"
    )
    if os.path.isfile(file):
        with open(file, encoding="utf8") as f:
            epoch_contributions[epoch] = json.load(f)

epoch_approximation_contributions = {}
for epoch in range(1000):
    file = os.path.join(
        save_dir,
        "approximation_hyper_gradient_contribution.epoch_" + str(epoch) + ".json",
    )
    if os.path.isfile(file):
        with open(file, encoding="utf8") as f:
            epoch_approximation_contributions[epoch] = json.load(f)

epoch_classic_influence_contributions_contributions = {}
for epoch in range(1000):
    file = os.path.join(
        save_dir, "classic_influence_function_contribution_" + str(epoch) + ".json"
    )
    if os.path.isfile(file):
        with open(file, encoding="utf8") as f:
            epoch_classic_influence_contributions_contributions[epoch] = json.load(f)


sign_fig, sign_ax = plt.subplots()
sign_ax.set_title("Different contribution sign rate")
sign_ax.set_ylabel("Rate")
relative_error_fig, relative_error_ax = plt.subplots()
relative_error_ax.set_title("Relative contribution error")
relative_error_ax.set_ylabel("Error")


hydra_sign_y = []
if_sign_y = []
hydra_error_y = []
if_error_y = []
for epoch in sorted(epoch_approximation_contributions.keys()):
    approximation_contribution = epoch_approximation_contributions[epoch]
    contribution = epoch_contributions[epoch]
    assert len(approximation_contribution) == len(contribution)
    total_len = len(contribution)
    values = []
    normal_indices = set()
    for idx in approximation_contribution.keys():
        if (
            approximation_contribution[idx] < 0
            and contribution[idx] >= 0
            or approximation_contribution[idx] >= 0
            and contribution[idx] < 0
        ):
            values.append(contribution[idx])
        else:
            normal_indices.add(idx)
    hydra_sign_y.append(len(values) / total_len)

    abs_relative_error = []
    for idx in sorted(normal_indices):
        if contribution[idx] == 0:
            continue
        abs_relative_error.append(
            abs(
                (approximation_contribution[idx] - contribution[idx])
                / contribution[idx]
            )
        )
    hydra_error_y.append(torch.Tensor(abs_relative_error).mean())

for epoch in sorted(epoch_classic_influence_contributions_contributions.keys()):
    influence_function_contribution = (
        epoch_classic_influence_contributions_contributions[epoch]
    )
    contribution = epoch_contributions[epoch]
    assert len(influence_function_contribution) == len(contribution)
    total_len = len(contribution)
    values = []
    normal_indices = set()
    for idx in approximation_contribution.keys():
        if influence_function_contribution[idx] * contribution[idx] < 0:
            values.append(contribution[idx])
        else:
            normal_indices.add(idx)
    if_sign_y.append(len(values) / total_len)
    abs_relative_error = []
    for idx in sorted(normal_indices):
        assert contribution[idx] * influence_function_contribution[idx] > 0
        abs_relative_error.append(
            abs(
                (influence_function_contribution[idx] - contribution[idx])
                / contribution[idx]
            )
        )
    if_error_y.append(torch.Tensor(abs_relative_error).mean())

sign_ax.plot(epochs, hydra_sign_y, label="HyDRA")
if if_sign_y:
    sign_ax.plot(epochs, if_sign_y, label="Influence Function")
sign_ax.legend()
sign_fig.savefig("sign_error.jpg")

relative_error_ax.plot(epochs, hydra_error_y, label="HyDRA")
if if_error_y:
    relative_error_ax.plot(epochs, if_error_y, label="Influence Function")
relative_error_ax.legend()
relative_error_fig.savefig("relative_error.jpg")


spearmanr_fig, spearmanr_ax = plt.subplots()
spearmanr_ax.set_title("Spearman's rank correlation coefficient")
spearmanr_ax.set_ylabel("Coefficient")

spearmanr_hydra = []
spearmanr_if = []

for epoch in sorted(epoch_approximation_contributions.keys()):
    contribution_approximation_vector = list(
        get_mapping_values_by_key_order(epoch_approximation_contributions[epoch])
    )
    contribution_vector = list(
        get_mapping_values_by_key_order(epoch_contributions[epoch])
    )

    correlation = stats.spearmanr(
        contribution_vector, contribution_approximation_vector
    ).correlation
    spearmanr_hydra.append(correlation)

for epoch in sorted(epoch_classic_influence_contributions_contributions.keys()):
    influence_function_contribution_vector = list(
        get_mapping_values_by_key_order(
            epoch_classic_influence_contributions_contributions[epoch]
        )
    )
    contribution_vector = list(
        get_mapping_values_by_key_order(epoch_contributions[epoch])
    )
    correlation = stats.spearmanr(
        contribution_vector, influence_function_contribution_vector
    ).correlation
    spearmanr_if.append(correlation)
spearmanr_ax.plot(epochs, spearmanr_hydra, label="HyDRA")
if spearmanr_if:
    spearmanr_ax.plot(epochs, spearmanr_if, label="Influence Function")
spearmanr_ax.legend()
spearmanr_fig.savefig("spearmanr.jpg")
