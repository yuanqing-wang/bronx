from typing import Optional
import torch
import pyro
import dgl
import gpytorch
from ...global_parameters import NUM_SAMPLES

class GraphMultiClassificationPyroSteps(object):
    @staticmethod
    def training_step(self, batch, batch_idx):
        """Training step for the model."""
        g, y = batch
        h = g.ndata["attr"]
        loss = self.svi.step(g, h, y)

        # NOTE: `self.optimizers` here is None
        # but this is to trick the lightning module
        # to count steps
        self.optimizers().step()
        return None
    
    @staticmethod
    def validation_step(self, batch, batch_idx):
        """Validation step for the model."""
        g, y = batch
        h = g.ndata["attr"]
        predictive = pyro.infer.Predictive(
            self.svi.model,
            guide=self.svi.guide,
            num_samples=NUM_SAMPLES,
            parallel=False,
            return_sites=["_RETURN"],
        )
        y_hat = predictive(g, h, y=None)["_RETURN"].mean(0)
        accuracy = (y_hat.argmax(-1) == y).float().mean()
        self.log("val/accuracy", accuracy)
        return accuracy
    
    @staticmethod
    def test_step(self, batch, batch_idx):
        """Validation step for the model."""
        g, y = batch
        h = g.ndata["attr"]
        predictive = pyro.infer.Predictive(
            self.svi.model,
            guide=self.svi.guide,
            num_samples=NUM_SAMPLES,
            parallel=False,
            return_sites=["_RETURN"],
        )

        y_hat = predictive(g, h, y=None)["_RETURN"].mean(0)
        accuracy = (y_hat.argmax(-1) == y).float().mean()
        self.log("test/accuracy", accuracy)
        return accuracy

class GraphMultiClassificationPyroHead(torch.nn.Module):
    steps = GraphMultiClassificationPyroSteps
    def forward(
            self, 
            g: dgl.DGLGraph, 
            h: torch.Tensor,
            y: Optional[torch.Tensor] = None,
            aggregator: str = "sum",
        ):
        aggregator = getattr(dgl, f"{aggregator}_nodes")
        g.ndata["h"] = h
        h = aggregator(g, "h")
        if y is not None:
            with pyro.plate("nodes", g.batch_size):
                return pyro.sample(
                    "y", 
                    pyro.distributions.Categorical(logits=h),
                    obs=y.squeeze(-1),
                )
        else:
            return h
        

class GraphMultiClassificationGPytorchSteps(object):
    @staticmethod
    def training_step(self, batch, batch_idx):
        """Training step for the model."""
        self.model.train()
        self.head.train()
        g, y = batch
        h = g.ndata["attr"]
        y_hat = self.model(g, h)
        loss = self.head.loss(y_hat, y)
        self.log("train/loss", loss)
        return loss
    
    @staticmethod
    def validation_step(self, batch, batch_idx):
        """Validation step for the model."""
        self.model.eval()
        self.head.eval()
        g, y = batch
        h = g.ndata["attr"]
        y_hat = self(g, h).probs.mean(0).argmax(-1)
        accuracy = (y == y_hat).float().mean()
        self.log("val/accuracy", accuracy)

class GraphMultiClassificationGPytorchHead(gpytorch.Module):
    aggregation = "sum"
    steps = GraphMultiClassificationGPytorchSteps
    def __init__(
            self, 
            num_classes: int,
            gp: gpytorch.models.VariationalGP,
            num_data: int,
            aggregator: str = "sum",
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

        self.aggregator = getattr(dgl, f"{aggregator}_nodes")

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