#!/usr/bin/env python3

import argparse
import json

import hydra
from cyy_naive_lib.log import get_logger
from cyy_torch_algorithm.hydra.hydra_config import HyDRAConfig

from util import analysis_contribution, save_image

config = HyDRAConfig()

other_config = None


@hydra.main(config_path="conf", version_base=None)
def load_config(conf):
    global config
    global other_config
    if len(conf) == 1:
        conf = next(iter(conf.values()))
    other_config = HyDRAConfig.load_config(config, conf, check_config=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--contribution_path", type=str, required=True)
    parser.add_argument("--threshold", type=float)
    load_config()
    trainer = config.create_trainer()

    with open(other_config["contribution_path"], mode="rt", encoding="utf8") as f:
        contribution_dict = {int(k): v for k, v in json.load(f).items()}

    positive_contributions, negative_contributions = analysis_contribution(
        contribution_dict, threshold=other_config["threshold"]
    )
    get_logger().info("positive contributions are %s", positive_contributions)
    get_logger().info("negative contributions are %s", negative_contributions)
    for k in positive_contributions:
        save_image(".", trainer, positive_contributions, index=k)

    for k in negative_contributions:
        save_image(".", trainer, negative_contributions, index=k)
    # analysis_result_dir = os.path.join(args.hydra_dir, "hydra_analysis_result")
    # if args.sample_index is not None:
    #     analysis_result_dir = os.path.join(
    #         analysis_result_dir, "sample_" + str(args.sample_index)
    #     )

    # mask = contribution > (max_contribution * args.threshold)
    # for idx in mask.nonzero().tolist():
    #     idx = idx[0]
    #     if args.sample_index is None:
    #         save_training_image(
    #             analysis_result_dir, tester, contribution, training_dataset, idx
    #         )

    # mask = contribution < (min_contribution * args.threshold)
    # for idx in mask.nonzero().tolist():
    #     idx = idx[0]
    #     if args.sample_index is None:
    #         save_training_image(
    #             analysis_result_dir, tester, contribution, training_dataset, idx
    #         )
