import torch
import dgl
import gpytorch
from .layer import GraphGPLayer

class UnwrappedFunctionalModel(gpytorch.Module):
    def __init__(
            self,
            layer: torch.nn.Module,
            in_features: int,
            out_features: int,
            hidden_features: int,
            depth: int,
            activation: torch.nn.Module = torch.nn.SiLU(),
            proj_in: bool = False,
            proj_out: bool = False,
    ):
        super().__init__()
        if proj_in:
            self.proj_in = torch.nn.Linear(in_features, hidden_features)
        if proj_out:
            self.proj_out = torch.nn.Linear(hidden_features, out_features)
        
        self.layers = torch.nn.ModuleList([
            GraphGPLayer(
                layer=layer,
                in_features=in_features if i == 0 else hidden_features,
                out_features=hidden_features if i < depth - 1 else out_features,
            )
            for i in range(depth)
        ])
