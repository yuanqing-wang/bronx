import os
from types import SimpleNamespace
import numpy as np
import torch
import lightning as pl
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.tune.search.optuna import OptunaSearch

CONFIG = {
    "depth": tune.randint(2, 32),
    "hidden_features": tune.lograndint(32, 128),
    "lr": tune.loguniform(1e-4, 1e-1),
    "weight_decay": tune.loguniform(1e-6, 1e-2),
    "batch_size": -1,
}

def train(config):
    from run import run
    config.update(args.__dict__)
    config["checkpoint"] = os.path.join(os.getcwd(), "checkpoint")
    config = SimpleNamespace(**config)
    run(config)

def run(args):
    # specify a scheduler
    scheduler = ASHAScheduler(
        max_t=args.num_epochs, 
        grace_period=args.grace_period, 
        reduction_factor=args.reduction_factor,
    )

    # specify the target
    target = tune.with_resources(train, {"cpu": 1, "gpu": 1})

    # specify the run configuration
    tuner = tune.Tuner(
            target,
            param_space=CONFIG,
            tune_config=tune.TuneConfig(
                metric="val/rmse",
                mode="min",
                num_samples=args.num_samples,
                scheduler=scheduler,
                search_alg=OptunaSearch(),
            ),
            run_config=RunConfig(
                    storage_path=os.path.join(
                        os.getcwd(), 
                        # ",".join(f"{k}={v}" for k, v in args.__dict__.items()),
                        f"{args.data}_{args.layer}_{args.strategy}",
                    ),
            ),
    )

    # execute the search
    tuner.fit()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="CoraGraphDataset")
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--grace_period", type=int, default=10)
    parser.add_argument("--reduction_factor", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--layer", type=str, default="GCN")

    # strategy-specific arguments
    subparsers = parser.add_subparsers(dest="strategy")

    # structural
    structural = subparsers.add_parser("structural")
    structural.add_argument("--head", type=str, default="GraphRegressionPyroHead")

    parametric = subparsers.add_parser("parametric")
    parametric.add_argument("--head", type=str, default="GraphRegressionPyroHead")

    functional = subparsers.add_parser("functional")
    functional.add_argument("--head", type=str, default="GraphRegressionGPytorchHead")
    
    node = subparsers.add_parser("node")
    node.add_argument("--head", type=str, default="GraphRegressionPyroHead")

    # parse args
    args = parser.parse_args()
    run(args)




