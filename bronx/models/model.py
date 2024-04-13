import abc
from typing import Optional
import torch
import dgl
import pyro
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
        loss = self.svi.step(*batch)

        # NOTE: `self.optimizers` here is None
        # but this is to trick the lightning module
        # to count steps
        self.optimizers().step()
        return None
    
    def validation_step(self, batch, batch_idx):
        """Validation step for the model."""
        g, h, y, mask = batch
        predictive = pyro.infer.Predictive(
            self.svi.model,
            guide=self.svi.guide,
            num_samples=1,
            parallel=False,
            return_sites=["_RETURN"],
        )

        y_hat = predictive(g, h, y=None, mask=None)["_RETURN"].mean(0)[mask]
        y = y[mask]
        accuracy = (y_hat.argmax(-1) == y).float().mean()
        self.log("val/accuracy", accuracy)
        return accuracy
    
    def test_step(self, batch, batch_idx):
        """Validation step for the model."""
        g, h, y, mask = batch
        predictive = pyro.infer.Predictive(
            self.svi.model,
            guide=self.svi.guide,
            num_samples=1,
            parallel=False,
            return_sites=["_RETURN"],
        )

        y_hat = predictive(g, h, y=None, mask=None)["_RETURN"].mean(0)[mask]
        y = y[mask]
        accuracy = (y_hat.argmax(-1) == y).float().mean()
        self.log("test/accuracy", accuracy)
        return accuracy




