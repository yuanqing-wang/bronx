import abc
import torch
from torch.utils.data import Dataset
from torch.distributions import Distribution
import lightning as pl

class BronxModel(abc.ABC):
    """Base class for all models in the Bronx framework.

    Methods
    -------
    __call__(self, *args, **kwargs)
        Make a prediction using the model.

    train(self, *args, **kwargs)
        Train the model using the given data.
    """
    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> Distribution:
        pass

    @abc.abstractmethod
    def train(self, dataset: Dataset) -> None:
        pass

class BronxLightningWrapper(pl.LightningModule):
    def __init__(self, model: BronxModel):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        loss = self.model(*batch)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters()


