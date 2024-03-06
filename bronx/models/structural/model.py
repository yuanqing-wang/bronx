import torch
from .layer import StructuralLayer
from dgl import DGLGraph

class StructuralModel(torch.nn.Module):
    """A model that characterizes the structural uncertainty of a graph.

    Parameters
    ----------
    
    """
    def __init__(
            self,
            in_features: int,
            out_features: int,
            hidden_features: int,
            edge_features: int,
            depth: int,
            activation: torch.nn.Module,
            proj_in: bool = False,
            proj_out: bool = False,
    ):
        super().__init__()


    def forward(
            self,
            g: DGLGraph,
            h: torch.Tensor,
    ):
        for layer in self.layers:
            h = layer(g, h)
        return h