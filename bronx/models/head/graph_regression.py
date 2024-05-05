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
            parallel=True,
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
    def forward(
            self, 
            g: dgl.DGLGraph, 
            h: torch.Tensor,
            y: Optional[torch.Tensor] = None,
        ):
        h_mu, h_log_sigma = h.chunk(2, dim=-1)
        h_sigma = torch.nn.functional.softplus(h_log_sigma)

        # if y is not None:
        h_mu, h_sigma = h_mu.squeeze(-1), h_sigma.squeeze(-1)
        if y is not None:
            y = y.squeeze(-1)
        with pyro.plate("nodes", g.batch_size):
            return pyro.sample(
                "y",
                pyro.distributions.Normal(h_mu, h_sigma), # .to_event(1),
                obs=y,
            )
        
class GraphRegressionGPytorchSteps(object):
    @staticmethod
    def training_step(self, batch, batch_idx):
        """Training step for the model."""
        _, g, y = batch
        h = g.ndata["h"]
        y = y.float()
        y_hat = self.model(g, h)
        loss = self.head.loss(y_hat, y)
        loss.backward()
        self.optimizers().step()
        return loss.item()
    
    @staticmethod
    def validation_step(self, batch, batch_idx):
        """Validation step for the model."""
        _, g, y = batch
        h = g.ndata["h"]
        y = y.float()
        y_hat = self.model(g, h)
        rmse = ((y_hat.mean - y) ** 2).mean().sqrt()
        self.log("val/rmse", rmse)
        return rmse
        

class GraphRegressionGPytorchHead(gpytorch.Module):
    steps = GraphRegressionGPytorchSteps
    def __init__(
            self, 
            gp: gpytorch.models.VariationalGP,
            num_data: int,
            **kwargs,
        ):
        super().__init__()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

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