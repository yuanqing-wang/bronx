import abc
from typing import Optional
import torch
import dgl
import pyro
from torch.utils.data import Dataset
from torch.distributions import Distribution
import lightning as pl
from ..global_parameters import NUM_SAMPLES

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
            *args,
            **kwargs,
    ):
        """Forward pass for the model."""
        h = self.model(g, h)
        return self.head(g, h, y, *args, **kwargs)
    
    def guide(
            self,
            g: dgl.DGLGraph,
            h: torch.Tensor,
            y: Optional[torch.Tensor],
            *args,
            **kwargs,
    ):
        """Guide pass for the model."""
        h = self.model.guide(g, h)
        return h
    
    def training_step(self, batch, batch_idx):
        """Training step for the model."""
        return self.head.steps.training_step(self, batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        """Validation step for the model."""
        return self.head.steps.validation_step(self, batch, batch_idx)
    
    def test_step(self, batch, batch_idx):
        """Test step for the model."""
        return self.head.steps.test_step(self, batch, batch_idx)
    




