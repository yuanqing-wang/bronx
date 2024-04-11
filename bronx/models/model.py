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

class BronxPyroMixin(object):
    """Base class for Pyro models in the Bronx framework."""
    def forward(
            self,
            g: dgl.DGLGraph,
            h: torch.Tensor,
            y: Optional[torch.Tensor],
            **kwargs,
    ):
        """Forward pass for the model."""
        h = self.model(g, h)
        return self.head(g, h, y=y, **kwargs)
    
    def guide(
            self,
            g: dgl.DGLGraph,
            h: torch.Tensor,
            y: Optional[torch.Tensor],
            **kwargs,
    ):
        """Guide pass for the model."""
        h = self.model.guide(g, h)
        return h

    def training_step(self, batch, batch_idx):
        """Training step for the model."""
        loss = self.svi.step(*batch)
        return loss



