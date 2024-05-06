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
        g = batch
        loss = self.svi.step(
            g,
            g.ndata["feat"],
            g.ndata["label"],
            g.ndata["train_mask"],
        )

        # NOTE: `self.optimizers` here is None
        # but this is to trick the lightning module
        # to count steps
        self.optimizers().step()
        return None
    
    @staticmethod
    def validation_step(self, batch, batch_idx):
        """Validation step for the model."""
        g = batch
        h = g.ndata["feat"]
        y = g.ndata["label"]
        mask = g.ndata["val_mask"]
        predictive = pyro.infer.Predictive(
            self.svi.model,
            guide=self.svi.guide,
            num_samples=NUM_SAMPLES,
            parallel=True,
            return_sites=["_RETURN"],
        )
        y_hat = predictive(g, h, y=None, mask=None)["_RETURN"]

        y_hat_vl = y_hat.mean(0)[g.ndata["val_mask"]]
        y_vl = y[g.ndata["val_mask"]]
        accuracy_vl = (y_hat_vl.argmax(-1) == y_vl).float().mean()
        self.log("val/accuracy", accuracy_vl, batch_size=1)

        y_hat_te = y_hat.mean(0)[g.ndata["test_mask"]]
        y_te = y[g.ndata["test_mask"]]
        accuracy_te = (y_hat_te.argmax(-1) == y_te).float().mean()
        self.log("test/accuracy", accuracy_te, batch_size=1)

        y_hat_te_samples = y_hat[:, g.ndata["test_mask"]]
        y_te_samples = y[g.ndata["test_mask"]].unsqueeze(0)
        accuracy_te_samples = (y_hat_te_samples.argmax(-1) == y_te_samples).float().mean(-1)
        self.log("test/accuracy_std", accuracy_te_samples.std(), batch_size=1)

        return accuracy_vl
    
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
            consistency_factor: float = 0.0,
    ):
        super().__init__()
        self.consistency_temperature = consistency_temperature
        self.consistency_factor = consistency_factor

        if self.consistency_factor > 0:
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

        if self.consistency_factor > 0:
            self.regularizer(h)

        # sample the logits
        if y is not None:
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
    
