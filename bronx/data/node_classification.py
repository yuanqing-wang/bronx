from functools import partial
import lightning as pl
import dgl

_ALL = [
    "CoraGraphDataset",
    "CiteseerGraphDataset",
    "PubmedGraphDataset",
    "CoauthorCSDataset",
    "CoauthorPhysicsDataset",
    "AmazonCoBuyComputerDataset",
    "AmazonCoBuyPhotoDataset",
    "FlickrDataset",
]

def _get_graph(data):
    assert data in _ALL, f"Data {data} not found in {_ALL}"
    g = getattr(dgl.data, data)()[0]
    g = dgl.to_bidirected(g, copy_ndata=True)
    return g

class SingleDataloader(object):
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        self.touched = False
        return self
    
    def __next__(self):
        if self.touched:
            raise StopIteration
        self.touched = True
        return self.data

class NodeClassificationDataModule(pl.LightningDataModule):
    def __init__(self, data):
        super().__init__()
        self.g = _get_graph(data)
        self.in_features = self.g.ndata["feat"].shape[-1]
        self.num_classes = self.g.ndata["label"].max().item() + 1
        
    def train_dataloader(self):
        return SingleDataloader([
            self.g,
            self.g.ndata["feat"],
            self.g.ndata["label"],
            self.g.ndata["train_mask"],
        ])
    
    def val_dataloader(self):
        return SingleDataloader([
            self.g,
            self.g.ndata["feat"],
            self.g.ndata["label"],
            self.g.ndata["val_mask"],
        ])

for data in _ALL:
    locals()[data] = partial(
        NodeClassificationDataModule,
        data=data,
    )

    





