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
    "CornellDataset",
    "TexasDataset",
    "WisconsinDataset",
]

def _get_graph(data, split=None):
    assert data in _ALL, f"Data {data} not found in {_ALL}"
    g = getattr(dgl.data, data)()[0]
    g = dgl.to_bidirected(g, copy_ndata=True)
    g = dgl.add_self_loop(g)
    if split is not None:
        g.ndata["train_mask"] = g.ndata["train_mask"][:, split]
        g.ndata["val_mask"] = g.ndata["val_mask"][:, split]
        g.ndata["test_mask"] = g.ndata["test_mask"][:, split]
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
    
def collate_fn(batch, mask_name="train_mask"):
    assert len(batch) == 1
    g = batch[0]
    # return (
    #     g,
    #     g.ndata["feat"],
    #     g.ndata["label"],
    #     g.ndata[mask_name],
    # )
    return g

class NodeClassificationDataModule(pl.LightningDataModule):
    def __init__(self, data, split=None):
        super().__init__()
        self.g = _get_graph(data, split=split)
        self.in_features = self.g.ndata["feat"].shape[-1]
        self.num_classes = self.g.ndata["label"].max().item() + 1
        
    def train_dataloader(self):
        return dgl.dataloading.GraphDataLoader(
            dataset=[self.g],
            batch_size=1,
            collate_fn=partial(collate_fn, mask_name="train_mask"),
        )
    
    def val_dataloader(self):
        return dgl.dataloading.GraphDataLoader(
            dataset=[self.g],
            batch_size=1,
            collate_fn=partial(collate_fn, mask_name="val_mask"),
        )
    
    def test_dataloader(self):
        return dgl.dataloading.GraphDataLoader(
            dataset=[self.g],
            batch_size=1,
            collate_fn=partial(collate_fn, mask_name="test_mask"),
        )

for data in _ALL:
    locals()[data] = partial(
        NodeClassificationDataModule,
        data=data,
    )

    





