import torch
import dgl
from dgl import DGLGraph

class EdgeModel(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
    ):
        super().__init__()
        self.fc = torch.nn.Linear(2*in_features, out_features)

    def forward(
            self,
            g: DGLGraph,
            h: torch.Tensor,
    ):
        g = g.local_var()
        g.ndata["h"] = h
        g.apply_edges(
            func=lambda edges: {
                "h": torch.cat([edges.src["h"], edges.dst["h"]], dim=-1)
            },
        )
        e = self.fc(g.edata["h"])
        return e

    