import argparse

from cyy_naive_pytorch_lib.algorithm.hydra.hydra_config import HyDRAConfig


class HyDRAExperimentConfig(HyDRAConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        super().optimizer_name = "SGD"
        self.label_noise_percent = None

    def load_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--label_noise_percent", type=float, required=True)
        super().load_args(parser=parser)


def get_config() -> HyDRAConfig:
    config = HyDRAExperimentConfig()
    config.load_args()
    return config
