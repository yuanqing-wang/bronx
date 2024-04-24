from typing import Optional
import torch
import dgl
import gpytorch
import lightning as pl
from .layer import GPLayer
from ...zoo.dgl import Sequential

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
            hidden_features: int,
            depth: int,
            activation: torch.nn.Module = torch.nn.SiLU(),
            proj_in: bool = False,
            *args, **kwargs,
    ):
        super().__init__()
        if proj_in:
            self.proj_in = torch.nn.Linear(in_features, hidden_features)
        
        self.layers = layer.sequential()(
            layer=layer,
            depth=depth,
            activation=activation,
            in_features=in_features,
            out_features=hidden_features,
            hidden_features=hidden_features,
        )

        self.gp = GPLayer(num_dim=hidden_features)
        self.activation = activation
        self.hidden_features = hidden_features

    def forward(
            self,
            g: dgl.DGLGraph,
            h: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ):
        if hasattr(self, "proj_in"):
            h = self.proj_in(h)
        h = self.layers(g, h)
        if mask is not None:
            h = h[..., mask, :]
        h = h.tanh()
        h = h.transpose(-1, -2).unsqueeze(-1)
        h = self.gp(h)
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
            out_features: int,
            num_data: int,
            lr: float = 1e-3,
            weight_decay: float = 1e-5,
            *args, **kwargs,
    ):
        super().__init__()
        self.model = UnwrappedFunctionalModel(*args, **kwargs)
        self.head = head(
            in_features=self.model.hidden_features,
            out_features=out_features,
            num_data=num_data,
            gp=self.model.gp,
        )
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(
            self,
            g: dgl.DGLGraph,
            h: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            *args,
            **kwargs,
    ):
        """Forward pass for the model."""
        h = self.model(g, h, mask=mask)
        return self.head(h, *args, **kwargs)
    
    def configure_optimizers(self):
        """Configure the optimizer for the model."""
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def training_step(self, batch, batch_idx):
        """Training step for the model."""
        self.model.train()
        self.head.train()
        g, h, y, mask = batch
        y_hat = self.model(g, h, mask=mask)
        if mask is not None:
            y = y[mask]
        loss = self.head.loss(y_hat, y)
        self.log("train/loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step for the model."""
        self.model.eval()
        self.head.eval()
        g, h, y, mask = batch
        y_hat = self(g, h, mask=mask).probs.mean(0).argmax(-1)
        if mask is not None:
            y = y[mask]
        accuracy = (y == y_hat).float().mean()
        self.log("val/accuracy", accuracy)

        