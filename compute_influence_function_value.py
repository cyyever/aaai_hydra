#!/usr/bin/env python3
import argparse
import json
import os

from cyy_torch_algorithm.influence_function import \
    compute_influence_function
from cyy_torch_algorithm.sample_gradient.sample_gradient_util import \
    get_sample_gradient_dict
from cyy_torch_toolbox.ml_type import MachineLearningPhase

from config import get_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_root_dir", type=str, required=True)
    parser.add_argument("--min_epoch", type=int)
    parser.add_argument("--max_epoch", type=int, default=1000)
    config = get_config(parser)

    trainer = config.create_trainer()
    inferencer = config.create_inferencer(phase=MachineLearningPhase.Test)
    tracking_indices_path = os.path.join(
        config.session_root_dir, "HyDRA", "tracking_indices.json"
    )
    if os.path.isfile(tracking_indices_path):
        with open(
            tracking_indices_path,
            mode="rt",
        ) as f:
            tracking_indices = json.load(f)
            print("use", len(tracking_indices), "indices")
    else:
        tracking_indices = list(range(len(trainer.dataset)))
    for epoch in range(config.min_epoch, config.max_epoch):
        model_path = os.path.join(
            config.session_root_dir, "model", "epoch_" + str(epoch) + ".pt"
        )
        if not os.path.isfile(model_path):
            continue
        print("compute in epoch", epoch)
        trainer.load_model(model_path)
        inferencer.load_model(model_path)
        test_gradient = inferencer.get_gradient()

        inferencer.transform_dataset(lambda _: trainer.dataset)

        training_sample_gradients = get_sample_gradient_dict(
            inferencer, tracking_indices
        )

        contributions = compute_influence_function(
            trainer,
            test_gradient,
            training_sample_gradients,
            batch_size=config.hyper_parameter_config.batch_size,
            dampling_term=0.01,
            scale=1000,
            epsilon=0.03,
        )
        with open(
            os.path.join(
                config.session_root_dir,
                "influence_function",
                "epoch_" + str(epoch) + ".json",
            ),
            mode="wt",
        ) as f:
            json.dump(contributions, f)
