import numpy as np
import pandas as pd
import pyro
from bronx.models import strategy
from bronx.models.head import graph_regression as heads
from bronx.models.zoo import dgl as zoo
from bronx.data.graph_regression import ESOL, FreeSolv, Lipophilicity
from bronx.active.acquisition import expected_improvement, probability_of_improvement
from bronx.active.experiment import experiment

def run(args):
    data = globals()[args.data]()
    data.setup()
    model = getattr(strategy, args.strategy.title() + "Model")(
        head=heads.GraphRegressionPyroHead,
        layer=getattr(zoo, args.layer),
        in_features=data.in_features,
        out_features=2,
        hidden_features=16,
        depth=2,
        num_data=len(data.data_train),
        autoguide=pyro.infer.autoguide.AutoDiagonalNormal,
        aggregation=True,
    )

    data = [(s, g, y) for s, g, y in data.data_train]
    portfolio = experiment(
        model,
        data,
        acquisition=globals()[args.acquisition],
        num_steps=args.num_steps,
        num_epochs=args.num_epochs,
    )
    scores = [y.item() for s, g, y in portfolio]
    scores = np.maximum.accumulate(scores).tolist()

    df = pd.DataFrame(columns=["data", "strategy", "acquisition", "layer", "score"])
    df["data"] = [args.data]
    df["strategy"] = [args.strategy]
    df["layer"] = [args.layer]
    df["score"] = [scores]
    df["acquisition"] = [args.acquisition]
    df.to_csv("results.csv", mode='a', header=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='ESOL')
    parser.add_argument('--strategy', type=str, default='Parametric')
    parser.add_argument('--layer', type=str, default='GCN')
    parser.add_argument('--acquisition', type=str, default='expected_improvement')
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--num_epochs', type=int, default=100)
    args = parser.parse_args()
    run(args)
