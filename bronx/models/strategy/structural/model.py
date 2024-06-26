from typing import Optional
from bronx.global_parameters import NUM_SAMPLES
import torch
import pyro
import dgl
from .layer import StructuralLayer
from .edge import EdgeLogitNormalPrior, EdgeLogitNormalGuide
from ...model import BronxLightningWrapper, BronxModel, BronxPyroMixin
from ...head import node_classification as node_classification_head
from ...zoo.dgl import Sequential
from ....global_parameters import NUM_SAMPLES
from dgl import DGLGraph
import lightning

class UnwrappedResampleStructuralModel(pyro.nn.PyroModule):
    """A model that characterizes the structural uncertainty of a graph.

    Parameters
    ----------
    in_features : int
        The number of input features.

    out_features : int
        The number of output features.

    hidden_features : int
        The number of hidden features.

    edge_features : int
        The number of edge features.

    prior : torch.nn.Module
        The prior distribution for the edge features.

    guide : torch.nn.Module
        The guide distribution for the edge features.

    depth : int
        The number of layers.

    activation : torch.nn.Module
        The activation function.

    proj_in : bool
        Whether to project the input features.

    proj_out : bool
        Whether to project the output features.

    Examples
    --------
    >>> import torch
    >>> import dgl
    >>> import bronx
    >>> from bronx.models.zoo.dgl import GCN
    >>> from bronx.models.structural.edge import EdgeLogitNormalPrior, EdgeLogitNormalGuide
    >>> g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
    >>> h = torch.randn(3, 10)
    >>> model = UnwrappedResampleStructuralModel(
    ...     layer=GCN,
    ...     in_features=10,
    ...     out_features=20,
    ...     hidden_features=15,
    ...     depth=2,
    ... )
    >>> h = model(g, h)
    >>> h.shape
    torch.Size([3, 20])
    """
    def __init__(
            self,
            layer: torch.nn.Module,
            in_features: int,
            out_features: int,
            hidden_features: int,
            depth: int,
            activation: torch.nn.Module = torch.nn.SiLU(),
            edge_features: int = 1,
            prior: torch.nn.Module = EdgeLogitNormalPrior,
            guide: torch.nn.Module = EdgeLogitNormalGuide,
            proj_in: bool = False,
            proj_out: bool = False,
            *args, **kwargs,
    ):
        super().__init__()
        if proj_in:
            self.proj_in = torch.nn.Linear(in_features, hidden_features)
        if proj_out:
            self.proj_out = torch.nn.Linear(hidden_features, out_features)
        self.layers = torch.nn.ModuleList([
            StructuralLayer(
                layer=layer,
                in_features=in_features if not proj_in and idx==0 else hidden_features,
                out_features=out_features if not proj_out and idx==depth-1 else hidden_features,
                edge_features=edge_features,
                prior=prior,
                guide=guide,
                edge_name=f"e{idx}",
            )
            for idx in range(depth)
        ])
        self.activation = activation

        
    def forward(
            self,
            g: DGLGraph,
            h: torch.Tensor,
    ):
        if hasattr(self, "proj_in"):
            h = self.proj_in(h)
        for layer in self.layers:
            h = layer(g, h)
            h = self.activation(h)
        if hasattr(self, "proj_out"):
            h = self.proj_out(h)
        return h
    
    def guide(
            self,
            g: DGLGraph,
            h: torch.Tensor,
    ):
        if hasattr(self, "proj_in"):
            h = self.proj_in(h)
        for layer in self.layers:
            h = layer.guide(g, h)
            h = self.activation(h)
        if hasattr(self, "proj_out"):
            h = self.proj_out(h)
        return h
    
class UnwrappedStructuralModel(pyro.nn.PyroModule):
    """A model that characterizes the structural uncertainty of a graph.

    Parameters
    ----------
    in_features : int
        The number of input features.

    out_features : int
        The number of output features.

    hidden_features : int
        The number of hidden features.

    edge_features : int
        The number of edge features.

    prior : torch.nn.Module
        The prior distribution for the edge features.

    guide : torch.nn.Module
        The guide distribution for the edge features.

    depth : int
        The number of layers.

    activation : torch.nn.Module
        The activation function.

    proj_in : bool
        Whether to project the input features.

    proj_out : bool
        Whether to project the output features.

    Examples
    --------
    >>> import torch
    >>> import dgl
    >>> import bronx
    >>> from bronx.models.zoo.dgl import GCN
    >>> from bronx.models.structural.edge import EdgeLogitNormalPrior, EdgeLogitNormalGuide
    >>> g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
    >>> h = torch.randn(3, 10)
    >>> model = UnwrappedStructuralModel(
    ...     layer=GCN,
    ...     in_features=10,
    ...     out_features=20,
    ...     hidden_features=15,
    ...     depth=2,
    ... )
    >>> h = model(g, h)
    >>> h.shape
    torch.Size([3, 20])
    """
    def __init__(
            self,
            layer: torch.nn.Module,
            in_features: int,
            out_features: int,
            hidden_features: int,
            depth: int,
            activation: torch.nn.Module = torch.nn.SiLU(),
            edge_features: int = 1,
            prior: torch.nn.Module = EdgeLogitNormalPrior,
            guide: torch.nn.Module = EdgeLogitNormalGuide,
            proj_in: bool = False,
            proj_out: bool = False,
            *args, **kwargs,
    ):
        super().__init__()
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
        )

        self.edge_module = StructuralLayer(
            layer=None,
            in_features=hidden_features if proj_in else in_features,
            edge_features=edge_features,
            prior=prior,
            guide=guide,
            edge_name="e",
        )

        self.activation = activation

        
    def forward(
            self,
            g: DGLGraph,
            h: torch.Tensor,
    ):
        if hasattr(self, "proj_in"):
            h = self.proj_in(h)
        e = self.edge_module(g, h)
        h = self.layers(g, h, edge_weight=e)
        if hasattr(self, "proj_out"):
            h = self.proj_out(h)
        return h
    
    def guide(
            self,
            g: DGLGraph,
            h: torch.Tensor,
    ):
        if hasattr(self, "proj_in"):
            h = self.proj_in(h)
        e = self.edge_module.guide(g, h)
        h = self.layers(g, h, edge_weight=e)
        if hasattr(self, "proj_out"):
            h = self.proj_out(h)
        return h

class StructuralModel(BronxPyroMixin, BronxLightningWrapper):
    """ Structural model wrapped in a lightning module.
    
    Examples
    --------
    >>> import torch
    >>> import dgl
    >>> import bronx
    >>> from bronx.models.zoo.dgl import GCN
    >>> from bronx.models.structural.edge import EdgeLogitNormalPrior, EdgeLogitNormalGuide
    >>> from bronx.models.head.node_classification import NodeClassificationPyroHead
    >>> g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
    >>> h = torch.randn(3, 10)
    >>> model = StructuralModel(
    ...     layer=GCN,
    ...     in_features=10,
    ...     out_features=20,
    ...     hidden_features=15,
    ...     depth=2,
    ...     head=NodeClassificationPyroHead,
    ... )
    """
    def __init__(
            self, 
            head: str,
            optimizer: str = "Adam",
            lr: float = 1e-2,
            weight_decay: float = 1e-3,
            loss: torch.nn.Module = pyro.infer.Trace_ELBO(),
            *args, 
            **kwargs
        ):
        model = UnwrappedStructuralModel(*args, **kwargs)
        super().__init__(model)

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
    
