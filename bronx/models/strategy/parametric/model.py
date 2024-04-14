import torch
import pyro
from .utils import init_log_sigma
from ..model import BronxModel
from ...zoo.dgl import Sequential

class UnwrappedParametricModel(pyro.nn.PyroModule):
    """ A model that characterizes the parametric uncertainty of a graph.
    
    Parameters
    ----------
    layer : torch.nn.Module
        The layer to use.

    in_features : int
        The number of input features.

    out_features : int
        The number of output features.

    hidden_features : int
        The number of hidden features.

    depth : int
        The number of layers.

    activation : torch.nn.Module
        The activation function.

    proj_in : bool
        Whether to project the input features.

    proj_out : bool
        Whether to project the output features.
    """

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
        
        self.layers = Sequential(
            layer=layer,
            depth=depth,
            activation=activation,
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
        )
