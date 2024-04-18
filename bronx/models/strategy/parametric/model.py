import torch
import pyro
import dgl
from .utils import init_sigma, to_pyro_module_
from ...zoo.dgl import Sequential
from ...model import BronxLightningWrapper, BronxModel, BronxPyroMixin

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
            *args, **kwargs,
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

    def forward(
            self,
            g: dgl.DGLGraph,
            h: torch.Tensor,
    ):
        if hasattr(self, "proj_in"):
            h = self.proj_in(h)
        h = self.layers(g, h)
        if hasattr(self, "proj_out"):
            h = self.proj_out(h)
        return h

class ParametricModel(BronxPyroMixin, BronxLightningWrapper):
    def __init__(
            self, 
            autoguide: pyro.infer.autoguide.guides.AutoGuide,
            head: str,
            sigma: float = 1.0,
            optimizer: str = "Adam",
            lr: float = 1e-2,
            weight_decay: float = 1e-3,
            loss: torch.nn.Module = pyro.infer.Trace_ELBO(),
            *args, 
            **kwargs,
    ):
        model = UnwrappedParametricModel(*args, **kwargs)
        to_pyro_module_(model)
        init_sigma(model, sigma)
        super().__init__(model)

        self.model.guide = autoguide(model)

        # initialize head
        self.head = head()
        self.automatic_optimization = False
        self.save_hyperparameters()

        # initialize optimizer
        optimizer = getattr(pyro.optim, optimizer)(
            {"lr": lr, "weight_decay": weight_decay},
        )

        self.svi = pyro.infer.SVI(
            self.forward,
            self.guide,
            optim=optimizer,
            loss=loss,
        )

    def configure_optimizers(self):
        return None
