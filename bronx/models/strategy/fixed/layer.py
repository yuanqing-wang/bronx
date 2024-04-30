import partial
import torch
import dgl

class VariationalDropOut(torch.nn.Module):
    """Dropout with variational inference. """
    def __init__(
            self,
            p: float = 0.5,
            *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dropout = partial(
            torch.nn.functional.dropout,
            p=p,
            training=True,
        )

    def forward(
            self,
            g: dgl.DGLGraph,
            h: torch.Tensor,
    ):
        h = self.dropout(h)
        return super().forward(g, h)