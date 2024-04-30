import torch
import dgl
import pyro
from dgl import function as fn

class GCN(pyro.nn.PyroModule):
    def __init__(self, in_features, out_features, suffix=""):
        super().__init__()
        self.W_mu = pyro.nn.PyroParam(torch.randn(in_features, out_features))
        self.W_log_sigma = pyro.nn.PyroParam(torch.randn(in_features, out_features))
        self.B = pyro.nn.PyroParam(torch.randn(out_features))
        self.suffix = suffix

    def forward(
            self,
            g: dgl.DGLGraph,
            h: torch.Tensor,
    ):
        """The forward pass. """
        W = pyro.sample(
            f"W_{self.suffix}",
            pyro.distributions.Normal(
                torch.zeros(g.number_of_nodes(), *self.W_mu.shape, device=h.device),
                torch.ones(g.number_of_nodes(), *self.W_mu.shape, device=h.device)
            ).to_event(2)
        )
        B = self.B
        return self._forward(g, h, W, B)
    
    def guide(
            self,
            g: dgl.DGLGraph,
            h: torch.Tensor,
    ):
        """The guide function. """
        W_mu = self.W_mu.expand(g.number_of_nodes(), *self.W_mu.shape)
        W_log_sigma = self.W_log_sigma.expand(g.number_of_nodes(), *self.W_log_sigma.shape)
        W = pyro.sample(
            f"W_{self.suffix}",
            pyro.distributions.Normal(
                W_mu,
                torch.exp(W_log_sigma)
            ).to_event(2)
        )
        B = self.B
        return self._forward(g, h, W, B)


    def _forward(
            self,
            g: dgl.DGLGraph,
            h: torch.Tensor,
            W: torch.Tensor,
            B: torch.Tensor,
    ):
        print(W.shape)
        # graph convolution
        degs = g.out_degrees().float().clamp(min=1).unsqueeze(-1)
        norm = torch.pow(degs, -0.5)
        h = h * norm
        h = torch.einsum(
            "...a,...ab->...b",
            h,
            W,
        )
        g.ndata["h"] = h
        g.update_all(
            message_func=fn.copy_u("h", "m"),
            reduce_func=fn.sum("m", "h"),
        )
        h = g.ndata["h"]
        h = h * norm
        return h


