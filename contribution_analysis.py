import json
import os

import hydra
import torch
from cyy_naive_lib.log import log_info
from cyy_torch_toolbox import Config
from cyy_torch_xai import SampleContributions
from cyy_torch_xai.hydra import HyDRAConfig

from util import analysis_contribution, save_image

config = HyDRAConfig()

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
    trainer = config.create_trainer()
    assert other_config is not None
    model_path = os.path.join(
        os.path.dirname(other_config["contribution_path"]),
        "..",
        "model",
        "last.pt",
    )
    trainer.model.load_state_dict(torch.load(model_path))

    with open(other_config["contribution_path"], encoding="utf8") as f:
        contribution_dict: SampleContributions = {
            int(k): v for k, v in json.load(f).items()
        }

        positive_contributions, negative_contributions = analysis_contribution(
            contribution_dict, threshold=other_config["threshold"]
        )
        log_info("positive contributions are %s", positive_contributions)
        log_info("negative contributions are %s", negative_contributions)
        for k in positive_contributions:
            save_image(".", trainer, positive_contributions, index=k)

        for k in negative_contributions:
            save_image(".", trainer, negative_contributions, index=k)
