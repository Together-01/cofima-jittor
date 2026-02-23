import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(filename)s] => %(message)s')
import argparse

import jittor as jt
jt.flags.use_cuda = 1

from method import CoFiMA, SequentialTrainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()     
    parser.add_argument("--method", type=str, default="cofima", choices=["sequential", "cofima"])
    cli_args = parser.parse_args()

    args = {
        # data_manager
        "dataset": "cifar100_224",
        "shuffle": True,
        "seed": 1993,
        "init_cls": 10,
        "increment": 10,
        "num_workers": 8,

        # model
        "model_name": "cofima_cifar_mocov3",
        "pretrained_path": "pretrained/mocov3-vit-base-300ep.pth",

        # parameters
        "batch_size": 32,
        "epochs": 15,
        "lr": 0.01,
        "weight_decay": 5e-4,
        "ca_epochs": 5,
        "ca_with_logit_norm": 0.05,
        "wt_alpha": 0.5,
        "init_w": -1,
        "fisher_weighting": True,
        "fisher_batch_size": 32,
        "milestones": [10],

        # device
        "device": "cuda" if jt.flags.use_cuda else "cpu"
    }

    logging.info(
        "effective cfg | model_name=%s dataset=%s epochs=%d ca_epochs=%d "
        "ca_with_logit_norm=%s milestones=%s wt_alpha=%s init_w=%s fisher_weighting=%s",
        args["model_name"],
        args["dataset"],
        args["epochs"],
        args["ca_epochs"],
        args["ca_with_logit_norm"],
        args["milestones"],
        args["wt_alpha"],
        args["init_w"],
        args["fisher_weighting"],
    )

    method_name = cli_args.method.lower()
    if method_name == "sequential":
        method = SequentialTrainer(args)
    elif method_name == "cofima":
        method = CoFiMA(args)
    else:
        raise ValueError("Unknown method: {}".format(method_name))
    
    method.run()


