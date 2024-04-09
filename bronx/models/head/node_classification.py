import torch
import pyro
import dgl

class NodeClassificationPyroHead(torch.nn.Module):
    def forward(
            self, 
            g: dgl.DGLGraph, 
            h: torch.Tensor,
        ):
        with pyro.plate("nodes", g.number_of_nodes()):
            return pyro.sample("y", pyro.distributions.Bernoulli(logits=h))