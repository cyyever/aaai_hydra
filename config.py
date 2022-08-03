import hydra
from cyy_torch_algorithm.hydra.hydra_config import HyDRAConfig
from cyy_torch_toolbox.default_config import DefaultConfig

global_config = HyDRAConfig()


@hydra.main(config_path="conf", version_base=None)
def load_config(conf):
    if len(conf) == 1:
        conf = next(iter(conf.values()))
    remain_config = HyDRAConfig.load_config(global_config, conf, check_config=True)
    assert not remain_config


def load_default_config(conf):
    if len(conf) == 1:
        conf = next(iter(conf.values()))
    remain_config = DefaultConfig.load_config(global_config, conf, check_config=True)
    assert not remain_config
