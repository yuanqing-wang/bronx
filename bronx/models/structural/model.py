import torch
from .layer import StructuralLayer
from dgl import DGLGraph

class StructuralModel(torch.nn.Module):
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
    >>> from bronx.models.structural.model import StructuralModel
    >>> from bronx.models.zoo.dgl import GCN
    >>> from bronx.models.structural.edge import EdgeLogitNormalPrior, EdgeLogitNormalGuide
    >>> g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
    >>> h = torch.randn(3, 10)
    >>> model = StructuralModel(
    ...     layer=GCN,
    ...     in_features=10,
    ...     out_features=20,
    ...     hidden_features=15,
    ...     edge_features=1,
    ...     prior=EdgeLogitNormalPrior,
    ...     guide=EdgeLogitNormalGuide,
    ...     depth=2,
    ...     activation=torch.nn.ReLU(),
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
            edge_features: int,
            prior: torch.nn.Module,
            guide: torch.nn.Module,
            depth: int,
            activation: torch.nn.Module,
            proj_in: bool = False,
            proj_out: bool = False,
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