import torch
import dgl
import pyro
import lightning as pl
from ray import train

from ray.tune.integration.pytorch_lightning import TuneReportCallback
class _TuneReportCallback(TuneReportCallback, pl.Callback):
    best = None

def run(args):
    from bronx.data import graph_classification
    data = getattr(graph_classification, args.data)()
    data.setup()
    import bronx.models.zoo.dgl as zoo
    from bronx.models import strategy
    from bronx.models.head import graph_binary_classification as heads
    model = getattr(strategy, args.strategy.title() + "Model")(
        head=getattr(heads, args.head),
        layer=getattr(zoo, args.layer),
        in_features=data.in_features,
        out_features=1,
        hidden_features=args.hidden_features,
        depth=args.depth,
        num_data=len(data.data_train),
        # autoguide=pyro.infer.autoguide.AutoDiagonalNormal,
        autoguide=pyro.infer.autoguide.AutoDelta,
        aggregation=True,
    )

    from lightning.pytorch.loggers import CSVLogger
    from lightning.pytorch.callbacks import ModelCheckpoint
    import lightning

    checkpoint_callback = ModelCheckpoint(
        monitor="val/accuracy",
        mode="max",
        verbose=False,
        dirpath=args.checkpoint,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
    )

    trainer = pl.Trainer(
        callbacks=[_TuneReportCallback(metrics="val/accuracy")],
        max_epochs=args.num_epochs, 
        accelerator="cpu",
        logger=CSVLogger("logs", name="structural"),
    )
    trainer.fit(model, data)
    
    # test
    if args.test:
        trainer.test(datamodule=data)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--test", type=int, default=0)

    # arguments shared by all programs
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--hidden_features", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--data", type=str, default="MUTAG")
    parser.add_argument("--layer", type=str, default="GCN")

    # strategy-specific arguments
    subparsers = parser.add_subparsers(dest="strategy")

    # structural
    structural = subparsers.add_parser("structural")
    structural.add_argument("--head", type=str, default="GraphBinaryClassificationPyroHead")

    # functional
    functional = subparsers.add_parser("functional")
    functional.add_argument("--head", type=str, default="GraphBinaryClassificationGPytorchHead")

    # parametric
    parametric = subparsers.add_parser("parametric")
    parametric.add_argument("--head", type=str, default="GraphBinaryClassificationPyroHead")

    node = subparsers.add_parser("node")
    node.add_argument("--head", type=str, default="GraphBinaryClassificationPyroHead")

    # parse arguments    
    args = parser.parse_args()
    print(args)
    run(args)
