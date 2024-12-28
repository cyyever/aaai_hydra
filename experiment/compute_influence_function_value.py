import json
import os

import hydra
import torch
from cyy_torch_toolbox import Config
from cyy_torch_xai.influence_function import compute_influence_function_values

config = Config()
other_config = None


@hydra.main(config_path="conf", version_base=None)
def load_config(conf):
    global config
    global other_config
    if len(conf) == 1:
        conf = next(iter(conf.values()))
    other_config = Config.load_config(config, conf, check_config=False)


if __name__ == "__main__":
    load_config()
    assert other_config is not None

    tracking_indices_path = os.path.join(
        other_config["session_root_dir"], "HyDRA", "tracking_indices.json"
    )
    if os.path.isfile(tracking_indices_path):
        with open(tracking_indices_path, encoding="utf8") as f:
            tracking_indices = json.load(f)
            print("use", len(tracking_indices), "indices")
    else:
        trainer = config.create_trainer()
        tracking_indices = list(range(len(trainer.dataset_util)))
    for epoch in range(1000):
        model_path = os.path.join(
            other_config["session_root_dir"], "model", "epoch_" + str(epoch) + ".pt"
        )
        if not os.path.isfile(model_path):
            continue
        print("compute in epoch", epoch)
        trainer = config.create_trainer()
        trainer.model.load_state_dict(torch.load(model_path))

        contributions = compute_influence_function_values(
            trainer=trainer,
            computed_indices=tracking_indices,
        )
        with open(
            os.path.join(
                other_config["session_root_dir"],
                "influence_function",
                "epoch_" + str(epoch) + ".json",
            ),
            encoding="utf8",
            mode="w",
        ) as f:
            json.dump(contributions, f)
