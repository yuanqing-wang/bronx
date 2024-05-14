from turtle import forward
import torch
import pyro
import dgl
from .node import NodeNormalPrior, NodeNormalGuide
from ..parametric.utils import to_pyro_module_
from ...zoo.dgl import Sequential
from ...model import BronxLightningWrapper, BronxModel, BronxPyroMixin
from ....global_parameters import NUM_SAMPLES


class UnwrappedStructuralModel(torch.nn.Module):
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
            aggregation: bool = False,
            prior: torch.nn.Module = NodeNormalPrior,
            guide: torch.nn.Module = NodeNormalGuide,
            *args, **kwargs,
    ):
        super().__init__()
        self.prior = prior()
        self._guide = guide(in_features=in_features)

        if proj_in:
            self.proj_in = torch.nn.Linear(in_features, hidden_features)
        if proj_out:
            self.proj_out = torch.nn.Linear(hidden_features, out_features)
        
        self.layers = layer.sequential()(
            layer=layer,
            depth=depth,
            activation=activation,
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            aggregation=aggregation,
        )

    def _mp(
            self,
            g: dgl.DGLGraph,
            h: torch.Tensor,
    ):
        g = g.local_var()
        if hasattr(self, "proj_in"):
            h = self.proj_in(h)
        h = self.layers(g, h)
        if hasattr(self, "proj_out"):
            h = self.proj_out(h)
        return h

    def forward(
            self,
            g: dgl.DGLGraph,
            h: torch.Tensor,
    ):
        # h = self.prior(g, h)
        h = self._mp(g, h)
        return h
    
    def guide(
            self,
            g: dgl.DGLGraph,
            h: torch.Tensor,
    ):
        # h = self._guide(g, h)
        return self._mp(g, h)
    
class StructuralModel(BronxPyroMixin, BronxLightningWrapper):
    def __init__(
            self, 
            head: str,
            optimizer: str = "Adam",
            lr: float = 1e-2,
            weight_decay: float = 1e-3,
            loss: torch.nn.Module = pyro.infer.Trace_ELBO(
                num_particles=NUM_SAMPLES,
                vectorize_particles=True,
            ),
            *args, 
            **kwargs,
    ):
        model = UnwrappedStructuralModel(*args, **kwargs)
        to_pyro_module_(model)
        super().__init__(model=model)
        
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
            num_samples=NUM_SAMPLES,
        )

    def configure_optimizers(self):
        return None
    
    # def forward(
    #         self,
    #         g: dgl.DGLGraph,
    #         h: torch.Tensor,
    #         y: torch.Tensor,
    #         mask: torch.Tensor,
    # ):
    #     h = self.model(g, h)
    #     return self.head(g, h, y=y, mask=mask)
    
    # def guide(
    #         self,
    #         g: dgl.DGLGraph,
    #         h: torch.Tensor,
    #         y: torch.Tensor,
    #         mask: torch.Tensor,
    # ):
    #     h = self.model.guide(g, h)
    #     return self.head(g, h, y=y, mask=mask)
    

