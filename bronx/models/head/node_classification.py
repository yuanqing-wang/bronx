from typing import Optional
import torch
import pyro
import dgl
import gpytorch

class NodeClassificationPyroHead(torch.nn.Module):
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
        

class NodeClassificationGPytorchHead(gpytorch.Module):
    def __init__(
            self, 
            in_features: int,
            out_features: int,
            gp: gpytorch.models.VariationalGP,
            num_data: int,
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