#!/usr/bin/env python3
import datetime
import json
import os

import torch
from cyy_naive_lib.log import set_file_handler
from cyy_torch_algorithm.hydra.hydra_config import HyDRAConfig
from cyy_torch_toolbox.hook import Hook
from cyy_torch_toolbox.ml_type import MachineLearningPhase

if __name__ == "__main__":
    config = HyDRAConfig()
    config.load_args()
    config.use_hessian = True
    config.use_approximation = True
    set_file_handler(
        os.path.join(
            "log",
            "hydra",
            config.dc_config.dataset_name,
            config.model_name,
            "{date:%Y-%m-%d_%H:%M:%S}.log".format(date=datetime.datetime.now()),
        )
    )
    hydra_trainer, hydra_hook = config.create_trainer(return_hydra_hook=True)

    class ComparisonHook(Hook):
        def _after_epoch(self, **kwargs):
            trainer = kwargs["model_executor"]
            epoch = kwargs["epoch"]
            save_dir = os.path.join(hydra_hook.save_dir, "approximation_comparision")
            if epoch % 4 != 0 and trainer.hyper_parameter.epoch != epoch:
                return
            hyper_gradient_distance = {}
            training_set_size = len(trainer.dataset)
            tester = trainer.get_inferencer(phase=MachineLearningPhase.Test)
            hessian_hyper_gradient_contribution = {}
            approximation_hyper_gradient_contribution = {}
            test_gradient = tester.get_gradient()

            def compute_approximation_contribution(index, hyper_gradient):
                approximation_hyper_gradient_contribution[index] = (
                    -(test_gradient @ hyper_gradient).data.item() / training_set_size
                )

            def compute_hessian_contribution(index, hyper_gradient):
                hessian_hyper_gradient_contribution[index] = (
                    -(test_gradient @ hyper_gradient).data.item() / training_set_size
                )

            hydra_hook.foreach_hyper_gradient(True, compute_approximation_contribution)
            hydra_hook.foreach_hyper_gradient(False, compute_hessian_contribution)

            def compute_distance(index, approx_hyper_gradient, hessian_hyper_gredient):
                hyper_gradient_distance[index] = torch.dist(
                    approx_hyper_gradient, hessian_hyper_gredient
                ).data.item()

            hydra_hook.foreach_approx_and_hessian_hyper_gradient(compute_distance)
            os.makedirs(save_dir, exist_ok=True)
            with open(
                os.path.join(
                    save_dir,
                    "approximation_distance_" + str(epoch) + ".json",
                ),
                mode="wt",
                encoding="utf8",
            ) as f:
                json.dump(hyper_gradient_distance, f)

            with open(
                os.path.join(
                    save_dir,
                    "approximation_hyper_gradient_contribution.epoch_"
                    + str(epoch)
                    + ".json",
                ),
                mode="wt",
                encoding="utf8",
            ) as f:
                json.dump(approximation_hyper_gradient_contribution, f)
            with open(
                os.path.join(
                    save_dir,
                    "hessian_hyper_gradient_contribution.epoch_" + str(epoch) + ".json",
                ),
                mode="wt",
                encoding="utf8",
            ) as f:
                json.dump(hessian_hyper_gradient_contribution, f)

    hydra_trainer.append_hook(ComparisonHook())
    hydra_trainer.train()
