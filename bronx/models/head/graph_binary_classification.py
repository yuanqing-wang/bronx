from typing import Optional
import torch
import pyro
import dgl
import gpytorch
from ...global_parameters import NUM_SAMPLES

class GraphBinaryClassificationPyroSteps(object):
    @staticmethod
    def training_step(self, batch, batch_idx):
        """Training step for the model."""
        g, y = batch
        h = g.ndata["attr"]
        y = y.unsqueeze(-1).float()
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
        y = y.unsqueeze(-1).float()
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
        y = y.unsqueeze(-1).float()
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

class GraphBinaryClassificationPyroHead(torch.nn.Module):
    steps = GraphBinaryClassificationPyroSteps
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
                    pyro.distributions.Bernoulli(logits=h.squeeze(-1)),
                    obs=y.squeeze(-1),
                )
        else:
            return h
        

class GraphBinaryClassificationGPytorchHead(gpytorch.Module):
    def __init__(
            self, 
            in_features: int,
            out_features: int,
            gp: gpytorch.models.VariationalGP,
            num_data: int,
            aggregator: str = "sum",
        ):
        super().__init__()
        self.likelihood = gpytorch.likelihoods.SoftmaxLikelihood(
            num_features=in_features,
            num_classes=out_features,
            mixing_weights=True,
        )

        self.mll = gpytorch.mlls.VariationalELBO(
            likelihood=self.likelihood,
            model=gp,
            num_data=num_data,
        )

        self.aggregator = getattr(dgl, f"{aggregator}_nodes")

    def forward(
            self,
            g: dgl.DGLGraph,
            h: torch.Tensor,
        ):
        g.ndata["h"] = h
        h = self.aggregator(g, "h")
        return self.likelihood(h)
    
    def loss(
            self,
            g: dgl.DGLGraph,
            h: torch.Tensor,
            y: torch.Tensor,
    ):
        g.ndata["h"] = h
        h = self.aggregator(g, "h")
        loss = -self.mll(h, y)
        return loss      