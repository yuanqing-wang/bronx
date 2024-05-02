from typing import Optional
import torch
import pyro
import dgl
import gpytorch
from ...global_parameters import NUM_SAMPLES

class GraphRegressionPyroSteps(object):
    @staticmethod
    def training_step(self, batch, batch_idx):
        """Training step for the model."""
        _, g, y = batch
        h = g.ndata["h"]
        y = y.float()
        loss = self.svi.step(g, h, y)

        # NOTE: `self.optimizers` here is None
        # but this is to trick the lightning module
        # to count steps
        self.optimizers().step()
        return None
    
    @staticmethod
    def validation_step(self, batch, batch_idx):
        """Validation step for the model."""
        _, g, y = batch
        h = g.ndata["h"]
        y = y.float()
        predictive = pyro.infer.Predictive(
            self.svi.model,
            guide=self.svi.guide,
            num_samples=NUM_SAMPLES,
            parallel=False,
            return_sites=["_RETURN"],
        )

        y_hat = predictive(g, h, y=None)["_RETURN"].mean(0)
        rmse = ((y_hat - y) ** 2).mean().sqrt()
        self.log("val/rmse", rmse)
        return rmse
    
    @staticmethod
    def test_step(self, batch, batch_idx):
        """Validation step for the model."""
        _, g, y = batch
        h = g.ndata["h"]
        y = y.float()
        predictive = pyro.infer.Predictive(
            self.svi.model,
            guide=self.svi.guide,
            num_samples=NUM_SAMPLES,
            parallel=False,
            return_sites=["_RETURN"],
        )

        y_hat = predictive(g, h, y=None)["_RETURN"].mean(0)
        rmse = ((y_hat - y) ** 2).mean().sqrt()
        self.log("test/rmse", rmse)
        return rmse

class GraphRegressionPyroHead(torch.nn.Module):
    steps = GraphRegressionPyroSteps
    aggregator = "mean"
    def __init__(
            self,
            in_features: int,
            out_features: int,
            activation: Optional[torch.nn.Module] = torch.nn.SiLU(),
    ):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features),
            activation,
            torch.nn.Linear(in_features, 2 * out_features),
        )

    def forward(
            self, 
            g: dgl.DGLGraph, 
            h: torch.Tensor,
            y: Optional[torch.Tensor] = None,
        ):
        aggregator = getattr(dgl, f"{self.aggregator}_nodes")
        parallel = h.shape[-1] != g.number_of_nodes()
        if parallel:
            h = h.swapaxes(-2, 0)
        g.ndata["h"] = h
        h = aggregator(g, "h")
        if parallel:
            h = h.swapaxes(0, -2)
        h = self.fc(h)
        h_mu, h_log_sigma = h.chunk(2, dim=-1)
        h_sigma = torch.nn.functional.softplus(h_log_sigma)
        if y is not None:
            with pyro.plate("nodes", g.batch_size):
                return pyro.sample(
                    "y", 
                    pyro.distributions.Normal(
                        h_mu,
                        h_sigma,
                    ).to_event(1),
                    obs=y,
                )
        else:
            return h
        

class GraphRegressionGPytorchHead(gpytorch.Module):
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