import datetime
import os

import cyy_torch_vision  # noqa: F401
import hydra
from cyy_naive_lib.log import add_file_handler
from cyy_torch_toolbox.hook.keep_model import KeepModelHook
from cyy_torch_xai.hydra import HyDRAConfig

config = HyDRAConfig()


@hydra.main(config_path="conf", version_base=None)
def load_config(conf) -> None:
    global config
    if len(conf) == 1:
        conf = next(iter(conf.values()))
    HyDRAConfig.load_config(config, conf, check_config=True)


if __name__ == "__main__":
    load_config()
    add_file_handler(
        os.path.join(
            "log",
            "hydra",
            config.dc_config.dataset_name,
            config.model_config.model_name,
            f"{datetime.datetime.now():%Y-%m-%d_%H:%M:%S}.log",
        )
    )
    hydra_trainer, _ = config.create_trainer_and_hook()
    model_hook = KeepModelHook()
    model_hook.save_last_model = True
    hydra_trainer.append_hook(model_hook)
    hydra_trainer.train()
