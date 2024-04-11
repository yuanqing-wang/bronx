import torch
import dgl
import lightning as pl

def run(args):
    from bronx.data import node_classification
    data = getattr(node_classification, args.data)()
    from bronx.models.zoo.dgl import GCN
    from bronx.models.structural.model import StructuralModel
    from bronx.models.head.node_classification import NodeClassificationPyroHead
    model = StructuralModel(
        head=NodeClassificationPyroHead(),
        layer=GCN,
        in_features=data.in_features,
        out_features=data.num_classes,
        hidden_features=args.hidden_features,
        depth=1,
    )

    from lightning.pytorch.loggers import CSVLogger
    trainer = pl.Trainer(
        max_epochs=100, 
        accelerator="cpu",
        logger=CSVLogger("logs", name="structural"),
        # check_val_every_n_epoch=1,
    )
    trainer.fit(model, data)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--hidden_features", type=int, default=64)
    parser.add_argument("--out_features", type=int, default=64)
    parser.add_argument("--edge_features", type=int, default=1)
    parser.add_argument("--data", type=str, default="CoraGraphDataset")
    args = parser.parse_args()
    run(args)
