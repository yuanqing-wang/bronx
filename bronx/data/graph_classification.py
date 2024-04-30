from functools import partial

from networkx import Graph
import torch
import lightning as pl
import dgl
from sklearn.model_selection import KFold

_ALL = [
    "MUTAG",
    "COLLAB",
    "IMDBBINARY",
    "IMDBMULTI",
    "NCI1",
    "PROTEINS",
    "PTC",
    "REDDITBINARY",
    "REDDITMULTI5K",
]

class GraphClassificationDataModule(pl.LightningDataModule):
    """ Graph classification datasets.

    Parameters
    ----------
    data : str
        Name of the dataset.

    seed : int
        Random seed.

    k : int
        Fold index.

    num_splits : int
        Number of splits.

    batch_size : int
        Batch size.

    Examples
    --------
    >>> data = GraphClassificationDataModule("MUTAG")
    >>> data.setup()
    >>> g, y = next(iter(data.train_dataloader()))
    >>> type(g)
    <class 'dgl.heterograph.DGLGraph'>
    >>> type(y)
    <class 'torch.Tensor'>
    >>> assert g.batch_size == y.size(0)
    """
    def __init__(
            self, 
            data: str,
            seed: int = 2666,
            k: int = 0,
            num_splits: int = 10,
            batch_size: int = -1,
        ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        self.data = dgl.data.GINDataset(
            self.hparams.data,
            self_loop=True,
        )
        kf = KFold(
            n_splits=self.hparams.num_splits, 
            shuffle=True, 
            random_state=self.hparams.seed,
        )
        all_splits = [k for k in kf.split(self.data)]
        train_indexes, val_indexes = all_splits[self.hparams.k]
        train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()
        self.data_train, self.data_val = (
            dgl.data.Subset(self.data, train_indexes),
            dgl.data.Subset(self.data, val_indexes),
        )
        if self.hparams.batch_size == -1:
            self.hparams.batch_size = len(self.data_train)
        g, y = self.data_train[0]
        self.in_features = g.ndata["attr"].size(-1)
        self.num_classes = y.max().item() + 1
        
    def train_dataloader(self):
        return dgl.dataloading.GraphDataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
        )
    
    def val_dataloader(self):
        return dgl.dataloading.GraphDataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
        )
    

for data in _ALL:
    locals()[data] = partial(
        GraphClassificationDataModule,
        data=data,
    )

    





