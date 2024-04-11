from typing import Optional
import torch
import pyro
import dgl

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
        else:
            number_of_nodes = g.number_of_nodes()

        with pyro.plate("obs_nodes", number_of_nodes):
            return pyro.sample(
                "y", 
                pyro.distributions.Categorical(logits=h[:, mask, ...]),
                obs=y[:, mask, ...],
            )