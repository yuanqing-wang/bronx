from audioop import bias
import torch
import pyro
import dgl
from dgl import DGLGraph

class NodeNormalPrior(torch.nn.Module):
    def __init__(
            self,
            name: str="h",
    ):
        super().__init__()
        self.name = name

    def forward(
            self,
            g: DGLGraph,
            h: torch.Tensor,
    ):
        super().__init__()
        with pyro.plate("nodes", g.number_of_nodes()):
            epsilon = pyro.sample(
                self.name,
                pyro.distributions.Normal(
                    torch.zeros(g.number_of_nodes(), 1, device=h.device),
                    torch.ones(g.number_of_nodes(), 1, device=h.device), 
                ).to_event(1),
            )
        # return h * (1 + 1e-3 * epsilon)
        return h
    
class NodeNormalGuide(torch.nn.Module):
    """Specify the normal guide distribution for the node features.

    """
    def __init__(
            self,
            in_features: int,
            name: str="h",
    ):
        super().__init__()
        self.in_features = in_features
        self.fc_mu = torch.nn.Linear(in_features, 1, bias=True)
        self.fc_log_sigma = torch.nn.Linear(in_features, 1, bias=True)
        self.name = name

    def forward(
            self,
            g: DGLGraph,
            h: torch.Tensor,
    ):
        g = g.local_var()
        mu, sigma = self.fc_mu(h), torch.exp(self.fc_log_sigma(h))
        with pyro.plate("nodes", g.number_of_nodes()):
            epsilon = pyro.sample(
                self.name,
                pyro.distributions.Normal(
                    mu,
                    sigma,
                ).to_event(1),
            )
        # return h * (1 + 1e-3 * epsilon)
        return h

    model = forward

        