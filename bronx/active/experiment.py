import torch
import dgl
import random
import lightning as pl
import os

def experiment(
        model,
        data,
        acquisition,
        num_steps,
        num_epochs,
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
        trainer = pl.Trainer(
            max_epochs=num_epochs, logger=False,
            default_root_dir=os.path.join(
                os.getcwd(), str(os.environ.get("SLURM_JOB_ID")),
            )
        )
        trainer.fit(model, data_train)
        _, g, y = next(iter(data))
        scores = acquisition(model, g, g.ndata["h"], best=y_best)
        best = scores.argmax()
        best = data.pop(best)
        portfolio.append(best)
    return portfolio





        
        

