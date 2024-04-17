import torch
from functools import partial
from dgl import DGLGraph

from dgl.nn import GraphConv
class GCN(GraphConv):
    """Graph Convolutional Networks. https://arxiv.org/abs/1609.02907

    Examples
    --------
    >>> import torch
    >>> import dgl
    >>> g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
    >>> h = torch.randn(3, 10)
    >>> model = GCN(10, 20)
    >>> h = model(g, h)
    >>> h.shape
    torch.Size([3, 20])
    """
    def __init__(self, *args, **kwargs):
        kwargs["allow_zero_in_degree"] = True
        super().__init__(*args, **kwargs)

    @property
    def in_features(self):
        return self._in_feats
    
    @property
    def out_features(self):
        return self._out_feats
    
from dgl.nn import SGConv
class SGC(SGConv):
    """Simplifying Graph Convolutional Networks. https://arxiv.org/abs/1902.07153

    Examples
    --------
    >>> import torch
    >>> import dgl
    >>> g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
    >>> h = torch.randn(3, 10)
    >>> model = SGC(10, 20)
    >>> h = model(g, h)
    >>> h.shape
    torch.Size([3, 20])
    """
    def __init__(self, *args, **kwargs):
        kwargs["allow_zero_in_degree"] = True
        super().__init__(*args, **kwargs)

    @property
    def in_features(self):
        return self._in_feats
    
    @property
    def out_features(self):
        return self._out_feats
    
from dgl.nn import GINConv
class GIN(GINConv):
    """Graph Isomorphism Networks. https://arxiv.org/abs/1810.00826

    Examples
    --------
    >>> import torch
    >>> import dgl
    >>> g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
    >>> h = torch.randn(3, 10)
    >>> model = GIN(10, 20)
    >>> h = model(g, h)
    >>> h.shape
    torch.Size([3, 20])
    """
    def __init__(self, in_feats, out_feats):
        lin = torch.nn.Linear(in_feats, out_feats)
        super().__init__(apply_func=lin, aggregator_type='sum')
        self._in_feats = in_feats
        self._out_feats = out_feats

    @property
    def in_features(self):
        return self._in_feats
    
    @property
    def out_features(self):
        return self._out_feats


class Sequential(torch.nn.Module):
    """A simple sequential model.

    Parameters
    ----------
    layer : torch.nn.Module
        The layer to use

    in_features : int
        The number of input features

    hidden_features : int
        The number of hidden features

    out_features : int
        The number of output features

    depth : int
        The number of layers

    activation : torch.nn.Module
        The activation function to use

    **kwargs
        Additional arguments to pass to the layer

    Examples
    --------
    >>> import torch
    >>> import dgl
    >>> import bronx
    >>> from bronx.models.zoo.dgl import Sequential
    >>> g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
    >>> h = torch.randn(3, 10)
    >>> model = bronx.models.zoo.dgl.Sequential(
    ...     GCN,
    ...     in_features=10,
    ...     hidden_features=20,
    ...     out_features=30,
    ...     depth=3,
    ...     activation=torch.nn.ReLU(),
    ... )
    >>> h = model(g, h)
    >>> h.shape
    torch.Size([3, 30])
    """
    def __init__(
            self,
            layer: torch.nn.Module,
            in_features: int,
            hidden_features: int,
            out_features: int,
            depth: int,
            activation: torch.nn.Module,
            **kwargs,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            layer(
                hidden_features if idx > 0 else in_features, 
                hidden_features if idx < depth - 1 else out_features, 
                **kwargs
            )
            for idx in range(depth)
        ])
        self.activation = activation

    def forward(
            self,
            g: DGLGraph,
            h: torch.Tensor,
            **kwargs,
    ):
        """Forward pass."""
        g = g.local_var()
        for idx, layer in enumerate(self.layers):
            h = layer(g, h, **kwargs)
            if idx < len(self.layers) - 1:
                h = self.activation(h)
        return h