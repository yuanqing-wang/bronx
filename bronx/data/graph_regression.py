from functools import partial
from networkx import Graph
import torch
import lightning as pl
import dgl
import dgllife
from sklearn.model_selection import KFold

_ALL = [
    "ESOL",
]

class GraphRegressionDataModule(pl.LightningDataModule):
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
    >>> data = GraphRegressionDataModule("ESOL")
    """
    def __init__(
            self, 
            data: str,
            seed: int = 2666,
            batch_size: int = -1,
        ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        from dgllife.utils import (
            CanonicalAtomFeaturizer,
            CanonicalBondFeaturizer,
        )
        self.data = getattr(dgllife.data, self.hparams.data)(
            node_featurizer=CanonicalAtomFeaturizer("h"),
            edge_featurizer=CanonicalBondFeaturizer("e"),

        )

        self.in_features = self.data[0][1].ndata["h"].shape[-1]

        from dgllife.utils import RandomSplitter
        splitter = RandomSplitter()
        self.data_train, self.data_valid, self.data_test = splitter.train_val_test_split(
            self.data, frac_train=0.8, frac_val=0.1, frac_test=0.1, 
        )



        if self.hparams.batch_size == -1:
            self.hparams.batch_size = len(self.data_train)

    def train_dataloader(self):
        return dgl.dataloading.GraphDataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
        )
    
    def val_dataloader(self):
        return dgl.dataloading.GraphDataLoader(
            dataset=self.data_valid,
            batch_size=self.hparams.batch_size,
        )
    

for data in _ALL:
    locals()[data] = partial(
        GraphRegressionDataModule,
        data=data,
    )

    





