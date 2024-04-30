from typing import Optional
import torch
import pyro
import dgl
import gpytorch

class GraphClassificationPyroHead(torch.nn.Module):
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
            return pyro.sample(
                "y", 
                pyro.distributions.Categorical(logits=h),
                obs=y,
            )
        else:
            return h
        

class GraphClassificationGPytorchHead(gpytorch.Module):
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