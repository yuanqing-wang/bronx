from typing import Optional
import torch
import pyro
import dgl

class StructuralLayer(torch.nn.Module):
    """Layer with structural uncertatinty with VI.

    Parameters
    ----------
    layer : torch.nn.Module
        The layer to use

    prior : torch.nn.Module
        The prior distribution for the edge features

    guide : torch.nn.Module
        The guide distribution for the edge features

    in_features : int
        The number of input features

    out_features : int
        The number of output features

    edge_features : int
        The number of edge features

    edge_name : str
        The name of the edge random variable

    Examples
    --------
    >>> import torch
    >>> import dgl
    >>> import bronx
    >>> from bronx.models.structural.layer import StructuralLayer
    >>> from bronx.models.zoo.dgl import GCN
    >>> from bronx.models.structural.edge import EdgeLogitNormalPrior, EdgeLogitNormalGuide
    >>> g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
    >>> h = torch.randn(3, 10)
    >>> layer = StructuralLayer(
    ...     GCN,
    ...     EdgeLogitNormalPrior,
    ...     EdgeLogitNormalGuide,
    ...     in_features=10,
    ...     out_features=20,
    ...     edge_features=1,
    ...     edge_name="e",
    ... )
    >>> h = layer(g, h)
    >>> h.shape
    torch.Size([3, 20])

    """
    def __init__(
            self,
            layer: Optional[torch.nn.Module],
            prior: torch.nn.Module,
            guide: torch.nn.Module,
            in_features: int,
            out_features: Optional[int]=None,
            edge_features: int=1,
            edge_name: str="e",
    ):
        super().__init__()
        if layer is not None:
            self.layer = layer(in_features, out_features)
        else:
            self.layer = None
        self.prior = prior(edge_features, name=edge_name)
        self._guide = guide(in_features, edge_features, name=edge_name)
        self.edge_name = edge_name

    def forward(
            self,
            g: dgl.DGLGraph,
            h: torch.Tensor,
    ):
        with pyro.plate("plate_" + self.edge_name, g.number_of_edges()):
            e = self.prior(g, h)
        if self.layer is not None:
            return self.layer(g, h, edge_weight=e)
        else:
            return e
    
    def guide(
            self,
            g: dgl.DGLGraph,
            h: torch.Tensor,
    ):
        with pyro.plate("plate_" + self.edge_name, g.number_of_edges()):
            e = self._guide(g, h)
        if self.layer is not None:
            return self.layer(g, h, edge_weight=e)
        else:
            return e


    