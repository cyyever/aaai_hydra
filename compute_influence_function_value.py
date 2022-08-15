#!/usr/bin/env python3
import argparse
import json
import os

import hydra
from cyy_torch_algorithm.influence_function_family.influence_function import \
    compute_influence_function
from cyy_torch_toolbox.default_config import DefaultConfig

config = DefaultConfig()


@hydra.main(config_path="conf", version_base=None)
def load_config(conf):
    global config
    if len(conf) == 1:
        conf = next(iter(conf.values()))
    DefaultConfig.load_config(config, conf, check_config=True)


if __name__ == "__main__":
    load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_root_dir", type=str, required=True)

    trainer = config.create_trainer()
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
    for epoch in range(1000):
        model_path = os.path.join(
            config.session_root_dir, "model", "epoch_" + str(epoch) + ".pt"
        )
        if not os.path.isfile(model_path):
            continue
        print("compute in epoch", epoch)
        trainer.load_model(model_path)

        contributions = compute_influence_function(
            trainer=trainer,
            computed_indices=tracking_indices,
        )
        with open(
            os.path.join(
                config.session_root_dir,
                "influence_function",
                "epoch_" + str(epoch) + ".json",
            ),
            encoding="utf8",
            mode="wt",
        ) as f:
            json.dump(contributions, f)
