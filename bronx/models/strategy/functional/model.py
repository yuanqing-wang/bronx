import torch
import dgl
import gpytorch
import lightning as pl
from .layer import GraphGPLayer

class UnwrappedFunctionalModel(gpytorch.Module):
    """A GP GNN model without the PyTorch Lightning wrapper.

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

    proj_in : bool
        Whether to project the input features.

    proj_out : bool
        Whether to project the output features.

    Examples
    --------
    >>> from bronx.models.zoo.dgl import GCN
    >>> model = UnwrappedFunctionalModel(
    ...     layer=GCN,
    ...     in_features=10,
    ...     out_features=5,
    ...     hidden_features=32,
    ...     depth=2,
    ... )
    >>> import dgl
    >>> import torch
    >>> g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
    >>> h = torch.randn(3, 10)
    >>> h = model(g, h)
    >>> h.shape
    torch.Size([3, 5])
    """
    def __init__(
            self,
            layer: torch.nn.Module,
            in_features: int,
            out_features: int,
            hidden_features: int,
            depth: int,
            activation: torch.nn.Module = torch.nn.Identity(),
            proj_in: bool = False,
            proj_out: bool = False,
    ):
        super().__init__()
        if proj_in:
            self.proj_in = torch.nn.Linear(in_features, hidden_features)
        if proj_out:
            self.proj_out = torch.nn.Linear(hidden_features, out_features)
        
        self.layers = torch.nn.ModuleList([
            GraphGPLayer(
                layer=layer,
                in_features=in_features if i == 0 else hidden_features,
                out_features=hidden_features if i < depth - 1 else out_features,
            )
            for i in range(depth)
        ])

        self.activation = activation

    def forward(
            self,
            g: dgl.DGLGraph,
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

class FunctionalModel(pl.LightningModule):
    """A GP GNN model with the PyTorch Lightning wrapper.

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

    proj_in : bool
        Whether to project the input features.

    proj_out : bool
        Whether to project the output features.

    Examples
    --------
    >>> from bronx.models.zoo.dgl import GCN
    >>> from bronx.models.head.node_classification import NodeClassificationGPytorchHead
    >>> model = FunctionalModel(
    ...     layer=GCN,
    ...     in_features=10,
    ...     out_features=5,
    ...     hidden_features=32,
    ...     depth=2,
    ... )
    >>> import dgl
    >>> import torch
    >>> g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
    >>> h = torch.randn(3, 10)
    >>> h = model(g, h)
    >>> h.shape
    torch.Size([3, 5])
    """
    def __init__(
            self,
            head: torch.nn.Module,
            lr: float = 1e-3,
            weight_decay: float = 1e-5,
            *args, **kwargs,
    ):
        super().__init__()
        self.model = UnwrappedFunctionalModel(*args, **kwargs)
        self.head = head

    def forward(
            self,
            g: dgl.DGLGraph,
            h: torch.Tensor,
            *args,
            **kwargs,
    ):
        """Forward pass for the model."""
        h = self.model(g, h)
        return self.head(h, *args, **kwargs)
    
    def configure_optimizers(self):
        """Configure the optimizer for the model."""
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
