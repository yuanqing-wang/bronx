import torch
import dgl
import pyro

class GCN(pyro.nn.PyroModule):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.W_mu = pyro.nn.PyroParam(torch.randn(in_feats, out_feats))
        self.W_log_sigma = pyro.nn.PyroParam(torch.randn(in_feats, out_feats))
        self.B = pyro.nn.PyroParam(torch.randn(out_feats))

    def forward(
            self,
            g: dgl.DGLGraph,
            h: torch.Tensor,
    ):
        """The forward pass. """
        W = pyro.sample(
            "W",
            pyro.distributions.Normal(
                torch.zeros(g.number_of_nodes(), *self.W_mu.shape),
                torch.ones(g.number_of_nodes(), *self.W_mu.shape)
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
        W_mu = self.W_mu
        W_log_sigma = self.W_log_sigma
        W = pyro.sample(
            "W",
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
        # graph convolution
        degs = g.out_degrees().float().clamp(min=1)
        norm = torch.pow(degs, -0.5)
        h = h * norm
        h = torch.einsum(
            "nab, nbc -> nac",
            h,
            W,
        )
        g.ndata["h"] = h
        g.update_all(
            message_func=dgl.function.copy_src(src="h", out="m"),
            reduce_func=dgl.function.sum(msg="m", out="h"),
        )
        h = g.ndata["h"]
        h = h * norm
        return h


