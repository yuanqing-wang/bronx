import torch
import pyro
import dgl
from dgl import DGLGraph

class EdgeConcatenation(torch.nn.Module):
    """Concatenates the source and destination node features to
    form the edge features.

    Parameters
    ----------
    in_features : int
        The number of input features

    out_features : int
        The number of output features

    Examples
    --------
    >>> import torch
    >>> import dgl
    >>> import bronx
    >>> from bronx.models.structural.edge import EdgeConcatenation
    >>> g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
    >>> h = torch.randn(3, 10)
    >>> model = EdgeConcatenation(10, 20)
    >>> e = model(g, h)
    >>> e.shape
    torch.Size([2, 20])
    """
    def __init__(
            self,
            in_features: int,
            out_features: int,
    ):
        super().__init__()
        self.fc_src = torch.nn.Linear(in_features, out_features, bias=False)
        self.fc_dst = torch.nn.Linear(in_features, out_features, bias=False)

    def forward(
            self,
            g: DGLGraph,
            h: torch.Tensor,
    ):
        g = g.local_var()
        h_src = self.fc_src(h)
        h_dst = self.fc_dst(h)
        g.ndata["h_src"], g.ndata["h_dst"] = h_src, h_dst
        g.apply_edges(
            func=lambda edges: {
                "e": edges.src["h_src"] + edges.dst["h_dst"],
            },
        )
        e = g.edata["e"]
        return e
    
class EdgeLogitNormalPrior(torch.nn.Module):
    """Specify the logit normal prior distribution for the edge features.

    Parameters
    ----------
    out_features : int
        The number of output features

    name : str
        The name of the random variable

    Examples
    --------
    >>> import torch
    >>> import dgl
    >>> import bronx
    >>> from bronx.models.strategy.structural.edge import EdgeLogitNormalPrior
    >>> g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
    >>> h = torch.randn(3, 10)
    >>> model = EdgeLogitNormalPrior(20, "e")
    >>> e = model(g, h)
    >>> e.shape
    torch.Size([2, 20])
    """
    def __init__(
            self,
            out_features: int,
            name: str="e",
    ):
        super().__init__()
        self.out_features = out_features
        self.name = name

    def forward(
            self,
            g: DGLGraph,
            h: torch.Tensor,
    ):
        g = g.local_var()
        e = pyro.sample(
            self.name,
            pyro.distributions.TransformedDistribution(
                pyro.distributions.Normal(
                    torch.zeros(g.number_of_edges(), self.out_features, device=h.device),
                    torch.ones(g.number_of_edges(), self.out_features, device=h.device),
                ),
                pyro.distributions.transforms.SigmoidTransform(),
            ).to_event(1),
        )
        return e
    
    model = forward

class EdgeLogitNormalGuide(torch.nn.Module):
    """Specify the logit normal guide distribution for the edge features.

    Parameters
    ----------
    in_features : int
        The number of input features

    out_features : int
        The number of output features

    name : str
        The name of the random variable

    Examples
    --------
    >>> import torch
    >>> import dgl
    >>> import bronx
    >>> from bronx.models.strategy.structural.edge import EdgeLogitNormalGuide
    >>> g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
    >>> h = torch.randn(3, 10)
    >>> model = EdgeLogitNormalGuide(10, 20, 1.0, name="e")
    >>> e = model(g, h)
    >>> e.shape
    torch.Size([2, 20])
    """
    def __init__(
            self,
            in_features: int,
            out_features: int,
            sigma_factor: float=1.0,
            name: str="e",
    ):
        super().__init__()
        self.edge_model = EdgeConcatenation(
            in_features=in_features,
            out_features=2*out_features,
        )
        self.sigma_factor = sigma_factor
        self.name = name

    def forward(
            self,
            g: DGLGraph,
            h: torch.Tensor,
    ):
        g = g.local_var()
        e = self.edge_model(g, h)
        e_mu, e_log_sigma = e.chunk(2, dim=-1)
        e = pyro.sample(
            self.name,
            pyro.distributions.TransformedDistribution(
                pyro.distributions.Normal(
                    e_mu,
                    self.sigma_factor * e_log_sigma.exp(),
                ),
                pyro.distributions.transforms.SigmoidTransform(),
            ).to_event(1),
        )
        return e
    
    guide = forward

    