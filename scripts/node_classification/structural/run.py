import torch
import dgl
import lightning as pl

from ray.tune.integration.pytorch_lightning import TuneReportCallback
class _TuneReportCallback(TuneReportCallback, pl.Callback):
    pass

def run(args):
    from bronx.data import node_classification
    data = getattr(node_classification, args.data)()
    from bronx.models.zoo.dgl import GCN
    from bronx.models.strategy.structural.model import StructuralModel
    from bronx.models.head.node_classification import NodeClassificationPyroHead
    model = StructuralModel(
        head=NodeClassificationPyroHead,
        layer=GCN,
        in_features=data.in_features,
        out_features=data.num_classes,
        hidden_features=args.hidden_features,
        depth=args.depth,
    )

    from lightning.pytorch.loggers import CSVLogger
    from lightning.pytorch.callbacks import ModelCheckpoint
    import lightning

    checkpoint_callback = ModelCheckpoint(
        monitor="val/accuracy",
        mode="max",
        verbose=True,
        dirpath="checkpoints",
        every_n_epochs=1,
        save_on_train_epoch_end=True,
    )

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, _TuneReportCallback()],
        max_epochs=10, 
        accelerator="cpu",
        logger=CSVLogger("logs", name="structural"),
    )
    trainer.fit(model, data)
    
    # # load best
    # model = StructuralModel.load_from_checkpoint(checkpoint_callback.best_model_path)
    # g, h, y, mask = next(iter(data.test_dataloader()))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--hidden_features", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--data", type=str, default="CoraGraphDataset")
    args = parser.parse_args()
    run(args)
