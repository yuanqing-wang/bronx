import torch
import pyro
import dgl
from .edge import EdgeModel

class StructuralLayer(torch.nn.Module):
    def __init__(
            self,
            layer: torch.nn.Module,
    ):
        super().__init__()
        self.layer = layer
        self.edge_model = EdgeModel(
            in_features=layer.in_features,
            out_features=layer.out_features,
        )

    def guide(
            self,
            g: dgl.DGLGraph,
            h: torch.Tensor,
    ):
        g = g.local_var()
        e = self.edge_model(g, h)        
        return e