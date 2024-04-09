import abc
import torch
from torch.utils.data import Dataset
from torch.distributions import Distribution
import lightning as pl

class BronxModel(torch.nn.Module):
    """Base class for all models in the Bronx framework.

    Methods
    -------
    __call__(self, *args, **kwargs)
        Make a prediction using the model.

    """
    pass

class BronxLightningWrapper(pl.LightningModule):
    def __init__(self, model: BronxModel):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        loss = self.model.loss(*batch)
        return loss



