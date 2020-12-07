#!/usr/bin/env python3
import datetime
import json
import os

import torch
from cyy_naive_lib.algorithm.sequence_op import split_list_to_chunks
from cyy_naive_lib.log import get_logger, set_file_handler
from cyy_naive_pytorch_lib.algorithm.influence_function.args import (
    add_arguments_to_parser, create_hyper_gradient_trainer_from_args)
from cyy_naive_pytorch_lib.arg_parse import get_parsed_args


def after_epoch_callback(hyper_gradient_trainer, epoch):
    save_dir = os.path.join(
        hyper_gradient_trainer.save_dir, "approximation_comparision"
    )
    trainer = hyper_gradient_trainer.trainer
    if epoch % 2 == 0 or trainer.get_hyper_parameter().epochs == epoch:
        trainer.save(save_dir, model_name="model_epoch_" + str(epoch) + ".pt")

    if epoch % 4 != 0 and trainer.get_hyper_parameter().epochs != epoch:
        return
    get_logger().info("begin do do_delayed_computation")
    hyper_gradient_trainer.do_delayed_computation()
    get_logger().info("end do do_delayed_computation")

    get_logger().info("begin comparison")

    test_gradient = trainer.get_validator(use_test_data=True).get_gradient()

    hyper_gradient_distance = dict()
    hessian_hyper_gradient_contribution = dict()
    approximation_hyper_gradient_contribution = dict()
    training_set_size = len(trainer.training_dataset)
    for chunk in split_list_to_chunks(
        hyper_gradient_trainer.hessian_hyper_gradient_mom_dict.keys(), 100
    ):
        hyper_gradient_trainer.hessian_hyper_gradient_mom_dict.prefetch(chunk)
        hyper_gradient_trainer.approx_hyper_gradient_mom_dict.prefetch(chunk)
        for index in chunk:
            hyper_gradient_distance[index] = torch.dist(
                hyper_gradient_trainer.get_hyper_gradient(index, True),
                hyper_gradient_trainer.get_hyper_gradient(index, False),
            ).data.item()
            hessian_hyper_gradient_contribution[index] = (
                -(
                    test_gradient
                    @ hyper_gradient_trainer.get_hyper_gradient(index, False)
                ).data.item()
                / training_set_size
            )
            approximation_hyper_gradient_contribution[index] = (
                -(
                    test_gradient
                    @ hyper_gradient_trainer.get_hyper_gradient(index, True)
                ).data.item()
                / training_set_size
            )

    get_logger().info("end comparison")
    os.makedirs(save_dir, exist_ok=True)
    with open(
        os.path.join(
            save_dir,
            "approximation_distance_" + str(epoch) + ".json",
        ),
        mode="wt",
    ) as f:
        json.dump(hyper_gradient_distance, f)

    with open(
        os.path.join(
            save_dir,
            "approximation_hyper_gradient_contribution.epoch_" +
            str(epoch) + ".json",
        ),
        mode="wt",
    ) as f:
        json.dump(approximation_hyper_gradient_contribution, f)
    with open(
        os.path.join(
            save_dir,
            "hessian_hyper_gradient_contribution.epoch_" +
            str(epoch) + ".json",
        ),
        mode="wt",
    ) as f:
        json.dump(hessian_hyper_gradient_contribution, f)
    # dump loss
    with open(
        os.path.join(save_dir, "training_loss.json"),
        mode="wt",
    ) as f:
        json.dump(trainer.training_loss, f)
    with open(
        os.path.join(save_dir, "validation_loss.json"),
        mode="wt",
    ) as f:
        json.dump(trainer.validation_loss, f)
    with open(
        os.path.join(save_dir, "validation_accuracy.json"),
        mode="wt",
    ) as f:
        json.dump(trainer.validation_accuracy, f)


if __name__ == "__main__":
    parser = add_arguments_to_parser()
    args = get_parsed_args(parser)

    set_file_handler(
        os.path.join(
            "log",
            "hypergradient_approximation",
            args.task_name,
            "{date:%Y-%m-%d_%H:%M:%S}.log".format(
                date=datetime.datetime.now()),
        )
    )

    args.use_hessian_and_approximation = True
    hyper_gradient_trainer = create_hyper_gradient_trainer_from_args(args)
    hyper_gradient_trainer.train(after_epoch_callbacks=[after_epoch_callback])
