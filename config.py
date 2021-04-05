from cyy_naive_pytorch_lib.algorithm.hydra.hydra_config import HyDRAConfig


class HyDRAExperimentConfig(HyDRAConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.optimizer_name = "SGD"
        # self.tracking_percentage = None
        self.tracking_percentage = 0.01


def get_config(parser=None) -> HyDRAExperimentConfig:
    config = HyDRAExperimentConfig()
    config.load_args(parser=parser)
    return config
