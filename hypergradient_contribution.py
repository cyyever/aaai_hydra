#!/usr/bin/env python3

import json
import os

from cyy_naive_pytorch_lib.algorithm.influence_function.args import \
    add_arguments_to_parser
from cyy_naive_pytorch_lib.algorithm.influence_function.hyper_gradient_analyzer import \
    HyperGradientAnalyzer
from cyy_naive_pytorch_lib.arg_parse import (get_inferencer_from_args,
                                             get_parsed_args)

if __name__ == "__main__":

    parser = add_arguments_to_parser()
    args = get_parsed_args(parser)

    result_dir = os.path.join(
        "hypergradient_contribution",
        args.task_name,
    )
    os.makedirs(
        result_dir,
        exist_ok=True,
    )

    tester = get_inferencer_from_args(args)
    analyzer = HyperGradientAnalyzer(tester, args.hyper_gradient_dir)

    test_subset = dict()
    for i in range(len(tester.dataset)):
        test_subset[i] = [i]

    contribution_dict = analyzer.get_training_sample_contributions(test_subset)

    with open(
        os.path.join(result_dir, "contribution.json"),
        mode="wt",
    ) as f:
        json.dump(contribution_dict, f)
