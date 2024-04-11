import torch
import dgl

def run(args):
    from dgl.data import (
        CoraGraphDataset,
        CiteseerGraphDataset,
        PubmedGraphDataset,
        CoauthorCSDataset,
        CoauthorPhysicsDataset,
        AmazonCoBuyComputerDataset,
        AmazonCoBuyPhotoDataset,
        CornellDataset,
        TexasDataset,
        FlickrDataset,
    )

    g = locals()[args.data](verbose=False)[0]
    g = dgl.remove_self_loop(g)
    g = dgl.to_bidirected(g, copy_ndata=True)

    from bronx.models.zoo.dgl import GCN
    from bronx.models.structural.model import StructuralModel
    from bronx.models.head.node_classification import NodeClassificationPyroHead
    from bronx.models.structural.model import StructuralModel
    model = StructuralModel(
        head=NodeClassificationPyroHead(),
        layer=GCN,
        in_features=g.ndata["feat"].shape[-1],
        out_features=g.ndata["label"].max().item() + 1,
        hidden_features=args.hidden_features,
        depth=1,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--hidden_features", type=int, default=64)
    parser.add_argument("--out_features", type=int, default=64)
    parser.add_argument("--edge_features", type=int, default=1)
