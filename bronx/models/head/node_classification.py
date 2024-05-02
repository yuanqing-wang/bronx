from re import template
from typing import Optional
import torch
import pyro
import dgl
import gpytorch
from ..regularizer import ConsistencyRegularizer

class NodeClassificationPyroSteps(object):
    @staticmethod
    def training_step(self, batch, batch_idx):
        """Training step for the model."""
        loss = self.svi.step(*batch)

        # NOTE: `self.optimizers` here is None
        # but this is to trick the lightning module
        # to count steps
        self.optimizers().step()
        return None
    
    @staticmethod
    def validation_step(self, batch, batch_idx):
        """Validation step for the model."""
        g, h, y, mask = batch
        predictive = pyro.infer.Predictive(
            self.svi.model,
            guide=self.svi.guide,
            num_samples=NUM_SAMPLES,
            parallel=True,
            return_sites=["_RETURN"],
        )

        y_hat = predictive(g, h, y=None, mask=None)["_RETURN"].mean(0)[mask]
        y = y[mask]
        accuracy = (y_hat.argmax(-1) == y).float().mean()
        self.log("val/accuracy", accuracy)
        return accuracy
    
    @staticmethod
    def test_step(self, batch, batch_idx):
        """Validation step for the model."""
        g, h, y, mask = batch
        predictive = pyro.infer.Predictive(
            self.svi.model,
            guide=self.svi.guide,
            num_samples=NUM_SAMPLES,
            parallel=True,
            return_sites=["_RETURN"],
        )

        y_hat = predictive(g, h, y=None, mask=None)["_RETURN"].mean(0)[mask]
        y = y[mask]
        accuracy = (y_hat.argmax(-1) == y).float().mean()
        self.log("test/accuracy", accuracy)
        return accuracy

class NodeClassificationPyroHead(torch.nn.Module):
    steps = NodeClassificationPyroSteps

    def __init__(
            self,
            consistency_temperature: float = 1.0,
            consistency_factor: float = 1.0,
    ):
        super().__init__()
        self.consistency_temperature = consistency_temperature
        self.consistency_factor = consistency_factor
        self.regularizer = ConsistencyRegularizer(
            temperature=self.consistency_temperature,
            factor=self.consistency_factor,
        )

    def forward(
            self, 
            g: dgl.DGLGraph, 
            h: torch.Tensor,
            y: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
        ):
        if mask is not None:
            number_of_nodes = mask.sum()
            h = h[..., mask, :]
            if y is not None:
                y = y[..., mask]
        else:
            number_of_nodes = g.number_of_nodes()

        # compute the scale
        scale = g.number_of_edges() / number_of_nodes

        # sample the logits
        if y is not None:
            with pyro.poutine.scale(None, scale=scale):
                with pyro.plate("obs_nodes", number_of_nodes):
                    return pyro.sample(
                        "y", 
                        pyro.distributions.Categorical(logits=h),
                        obs=y,
                    )
        else:
            return h
    
class NodeClassificationGPytorchSteps(object):
    @staticmethod
    def training_step(self, batch, batch_idx):
        """Training step for the model."""
        self.model.train()
        self.head.train()
        g, h, y, mask = batch
        y_hat = self.model(g, h, mask=mask)
        if mask is not None:
            y = y[mask]
        loss = self.head.loss(y_hat, y)
        self.log("train/loss", loss)
        return loss
    
    @staticmethod
    def validation_step(self, batch, batch_idx):
        """Validation step for the model."""
        self.model.eval()
        self.head.eval()
        g, h, y, mask = batch
        y_hat = self(g, h, mask=mask).probs.mean(0).argmax(-1)
        if mask is not None:
            y = y[mask]
        accuracy = (y == y_hat).float().mean()
        self.log("val/accuracy", accuracy)

from ...global_parameters import NUM_SAMPLES
class NodeClassificationGPytorchHead(gpytorch.Module):
    steps = NodeClassificationGPytorchSteps
    aggregation = None
    def __init__(
            self, 
            num_classes: int,
            gp: gpytorch.models.VariationalGP,
            num_data: int,
        ):
        super().__init__()
        self.likelihood = gpytorch.likelihoods.SoftmaxLikelihood(
            num_classes=num_classes,
            mixing_weights=None,
        )

        self.mll = gpytorch.mlls.VariationalELBO(
            likelihood=self.likelihood,
            model=gp,
            num_data=num_data,
        )

    def forward(
            self,
            h: torch.Tensor,
        ):
        return self.likelihood(h)
    
    def loss(
            self,
            h: torch.Tensor,
            y: torch.Tensor,
    ):
        loss = -self.mll(h, y)
        return loss      
    
