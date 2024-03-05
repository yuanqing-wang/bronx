import torch
import pyro
import dgl
from dgl import DGLGraph

class EdgeConcatenation(torch.nn.Module):
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
                "h": edges.src["h_src"] + edges.dst["h_dst"],
            },
        )
        e = self.fc(g.edata["h"])
        return e
    
class EdgeLogitNormalPrior(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        name = kwargs.pop("name", None)
        super().__init__(*args, **kwargs)

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
                    torch.zeros(g.number_of_edges(), 1),
                    torch.ones(g.number_of_edges(), 1),
                ),
                pyro.distributions.transforms.SigmoidTransform(),
            ).to_event(2),
        )
    
    model = forward

class EdgeLogitNormalGuide(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            name: str="e",
    ):
        super().__init__()
        self.edge_model = EdgeConcatenation(
            in_features=in_features,
            out_features=2*out_features,
        )
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
            ).to_event(2),
        )
        return e
    
    guide = forward

    