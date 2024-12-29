import datetime
import os

import cyy_torch_vision  # noqa: F401
import hydra as config_loader
from cyy_naive_lib.log import add_file_handler
from cyy_torch_toolbox.hook.keep_model import KeepModelHook
from cyy_torch_xai.lean_hydra import LeanHyDRAConfig

config = LeanHyDRAConfig()


@config_loader.main(config_path="conf", version_base=None)
def load_config(conf) -> None:
    global config
    if len(conf) == 1:
        conf = next(iter(conf.values()))
    LeanHyDRAConfig.load_config(config, conf, check_config=False)


if __name__ == "__main__":
    load_config()
    add_file_handler(
        os.path.join(
            "log",
            config.dc_config.dataset_name,
            config.model_config.model_name,
            f"{datetime.datetime.now():%Y-%m-%d_%H_%M_%S}.log",
        )
    )
    lean_hydra_trainer = config.create_deterministic_trainer()
    model_hook = KeepModelHook()
    model_hook.save_last_model = True
    lean_hydra_trainer.append_hook(model_hook)
    lean_hydra_trainer.train()
    lean_hydra_trainer, _ = config.recreate_trainer_and_hook()
    lean_hydra_trainer.train()
