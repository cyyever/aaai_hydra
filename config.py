from cyy_torch_algorithm.hydra.hydra_config import HyDRAConfig


def get_config(parser=None) -> HyDRAConfig:
    config = HyDRAConfig()
    config.hyper_parameter_config.optimizer_name = "SGD"
    config.load_args(parser=parser)
    return config
