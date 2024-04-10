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
        ):
        with pyro.plate("obs_nodes", g.number_of_nodes()):
            return pyro.sample(
                "y", 
                pyro.distributions.Categorical(logits=h),
                obs=y,
            )