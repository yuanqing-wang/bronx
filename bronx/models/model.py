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
    
    @classmethod
    def custom_load_from_checkpoint(cls, checkpoint_path: str):
        """Load a model from a checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        hyper_parameters = checkpoint["hyper_parameters"]
        model = cls(**hyper_parameters)
        state_dict = checkpoint["state_dict"]
        state_dict_without_guide = {
            key: value for key, value in state_dict.items() if "guide" not in key
        }
        model.load_state_dict(state_dict_without_guide)

        print(state_dict["model.guide.loc"])
        model.model.guide.loc = state_dict["model.guide.loc"]
        model.model.guide.scale_unconstrained = state_dict["model.guide.scale_unconstrained"]
        return model
    
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
    




