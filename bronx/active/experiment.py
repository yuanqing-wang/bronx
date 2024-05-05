import torch
import dgl
import random
import lightning as pl

def experiment(
        model,
        data,
        acquisition,
        num_steps,
):
    best = random.randint(0, len(data))
    first = data.pop(best)
    portfolio = [first]
    for step in range(num_steps):
        y_best = max([y] for _, g, y in portfolio)
        y_best = torch.tensor(y_best)
        data_train = dgl.dataloading.GraphDataLoader(
            dataset=portfolio,
            batch_size=len(portfolio),
        )
        trainer = pl.Trainer(max_epochs=10)
        trainer.fit(model, data_train)
        _, g, y = next(iter(data))
        scores = acquisition(model, g, g.ndata["h"], best=y_best)
        best = scores.argmax()
        best = data.pop(best)
        portfolio.append(best)
    return portfolio

if __name__ == "__main__":
    import pyro
    from bronx.models import strategy
    from bronx.models.head import graph_regression as heads
    from bronx.models.zoo import dgl as zoo
    from bronx.data.graph_regression import ESOL
    from bronx.active.acquisition import expected_improvement, probability_of_improvement
    data = ESOL()
    data.setup()
    model = strategy.NodeModel(
        head=heads.GraphRegressionPyroHead,
        layer=zoo.GCN,
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
        acquisition=expected_improvement,
        num_steps=10,
    )



        
        

