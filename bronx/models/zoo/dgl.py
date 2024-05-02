from re import L
import torch
from functools import partial
import dgl
from dgl import DGLGraph

class GCN(torch.nn.Module):
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
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = torch.nn.Linear(in_features, out_features)
    
    def forward(
            self,
            g: dgl.DGLGraph,
            h: torch.Tensor,
    ):
        g = g.local_var()
        norm = torch.pow(g.in_degrees().float().clamp(min=1), -0.5).to(h.device).unsqueeze(1)
        h = h * norm
        h = h @ self.fc.weight.swapaxes(-1, -2)
        parallel = h.shape[0] != g.number_of_nodes()
        
        if parallel:
            h = h.swapaxes(0, -2)
        g.ndata["h"] = h
        g.update_all(
            dgl.function.copy_u("h", "m"),
            dgl.function.sum("m", "h"),
        )
        h = g.ndata.pop("h")
        if parallel:
            h = h.swapaxes(0, -2)
        h = h * norm
        return h


    @classmethod
    def sequential(cls):
        return Sequential
    
class GAT(torch.nn.Module):
    """Graph Attention Networks. https://arxiv.org/abs/1710.10903

    """
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self._in_feats, self._out_feats = in_feats, out_feats
        self.fc = torch.nn.Linear(in_feats, out_feats, bias=False)
        self.fc_edge_left = torch.nn.Linear(out_feats, 1, bias=False)
        self.fc_edge_right = torch.nn.Linear(out_feats, 1, bias=False)

    @property
    def in_features(self):
        return self._in_feats
    
    @property
    def out_features(self):
        return self._out_feats
    
    def forward(
            self,
            g: dgl.DGLGraph,
            h: torch.Tensor,
    ):
        """Forward pass."""
        g = g.local_var()
        h = h @ self.fc.weight.swapaxes(-1, -2)
        e_left, e_right = h @ self.fc_edge_left.weight.swapaxes(-1, -2), h @ self.fc_edge_right.weight.swapaxes(-1, -2)

        parallel = h.shape[0] != g.number_of_nodes()
        if parallel:
            h = h.swapaxes(0, -2)
            e_left = e_left.swapaxes(0, -2)
            e_right = e_right.swapaxes(0, -2)
        
        g.ndata["h"] = h
        g.ndata["e_left"], g.ndata["e_right"] = e_left, e_right
        g.apply_edges(dgl.function.u_add_v("e_left", "e_right", "e"))
        g.edata["e"] = torch.nn.functional.leaky_relu(g.edata["e"], negative_slope=0.2)
        from dgl.nn.functional import edge_softmax
        e = edge_softmax(g, g.edata["e"])
        g.edata["e"] = e
        g.update_all(
            dgl.function.u_mul_e("h", "e", "m"),
            dgl.function.sum("m", "h"),
        )
        h = g.ndata.pop("h")
        if parallel:
            h = h.swapaxes(0, -2)
        return h

    @classmethod
    def sequential(cls):
        return Sequential
    
from dgl.nn import SGConv
class SGC(dgl.nn.SGConv):
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
    
    def forward(self, graph, feat, edge_weight=None):
        r"""

        Description
        -----------
        Compute Simplifying Graph Convolution layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.
        edge_weight: torch.Tensor, optional
            edge_weight to use in the message passing process. This is equivalent to
            using weighted adjacency matrix in the equation above, and
            :math:`\tilde{D}^{-1/2}\tilde{A} \tilde{D}^{-1/2}`
            is based on :class:`dgl.nn.pytorch.conv.graphconv.EdgeWeightNorm`.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.

        Note
        ----
        If ``cache`` is set to True, ``feat`` and ``graph`` should not change during
        training, or you will get wrong results.
        """
        import torch as th
        from torch import nn

        from dgl import function as fn
        from dgl.base import DGLError
        from dgl.nn.pytorch.conv.graphconv import EdgeWeightNorm
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

            msg_func = fn.copy_u("h", "m")
            if edge_weight is not None:
                # graph.edata["_edge_weight"] = EdgeWeightNorm("both")(
                #     graph, edge_weight
                # )
                graph.edata["_edge_weight"] = edge_weight
                msg_func = fn.u_mul_e("h", "_edge_weight", "m")

            if self._cached_h is not None:
                feat = self._cached_h
            else:
                if edge_weight is None:
                    # compute normalization
                    degs = graph.in_degrees().to(feat).clamp(min=1)
                    norm = th.pow(degs, -0.5)
                    norm = norm.to(feat.device).unsqueeze(1)
                # compute (D^-1 A^k D)^k X
                for _ in range(self._k):
                    if edge_weight is None:
                        feat = feat * norm
                    graph.ndata["h"] = feat
                    graph.update_all(msg_func, fn.sum("m", "h"))
                    feat = graph.ndata.pop("h")
                    if edge_weight is None:
                        feat = feat * norm

                if self.norm is not None:
                    feat = self.norm(feat)

                # cache feature
                if self._cached:
                    self._cached_h = feat
            return self.fc(feat)

    @classmethod
    def sequential(cls):
        return Sequential

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
    
    @classmethod
    def sequential(cls):
        return Sequential
    
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
            parallel = h.shape[0] != g.number_of_nodes()
            h = layer(g, h, **kwargs)
            if idx < len(self.layers) - 1:
                h = self.activation(h)
        return h
    
from dgl.nn import GCN2Conv
class GCNII(GCN2Conv):
    """Simple implementation of GCNII. https://arxiv.org/abs/2007.02133

    Examples
    --------
    >>> import torch
    >>> import dgl
    >>> g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
    >>> h = torch.randn(3, 10)
    >>> model = GCNII(10, 20)
    >>> h = model(g, h)
    >>> h.shape
    torch.Size([3, 20])
    """
    def __init__(self, in_feats, layer):
        super().__init__(
            in_feats, layer=layer, project_initial_features=True,
        )

    @property
    def in_features(self):
        return self._in_feats
    
    @property
    def out_features(self):
        return self._out_feats
    
    @classmethod
    def sequential(cls):
        return SequentialII
    
    def forward(self, graph, feat, feat_0, edge_weight=None):
        r"""

        Description
        -----------
        Compute graph convolution.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is the size of input feature and :math:`N` is the number of nodes.
        feat_0 : torch.Tensor
            The initial feature of shape :math:`(N, D_{in})`
        edge_weight: torch.Tensor, optional
            edge_weight to use in the message passing process. This is equivalent to
            using weighted adjacency matrix in the equation above, and
            :math:`\tilde{D}^{-1/2}\tilde{A} \tilde{D}^{-1/2}`
            is based on :class:`dgl.nn.pytorch.conv.graphconv.EdgeWeightNorm`.


        Returns
        -------
        torch.Tensor
            The output feature

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.

        Note
        ----
        * Input shape: :math:`(N, *, \text{in_feats})` where * means any number of additional
          dimensions, :math:`N` is the number of nodes.
        * Output shape: :math:`(N, *, \text{out_feats})` where all but the last dimension are
          the same shape as the input.
        * Weight shape: :math:`(\text{in_feats}, \text{out_feats})`.
        """
        import torch as th
        from torch import nn

        from dgl import function as fn
        from dgl.base import DGLError
        from dgl.nn.pytorch.conv.graphconv import EdgeWeightNorm

        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

            # normalize  to get smoothed representation
            if edge_weight is None:
                degs = graph.in_degrees().float().clamp(min=1)
                norm = th.pow(degs, -0.5)
                norm = norm.to(feat.device).unsqueeze(1)

            if edge_weight is None:
                feat = feat * norm
            graph.ndata["h"] = feat
            msg_func = fn.copy_u("h", "m")
            if edge_weight is not None:
                graph.edata["_edge_weight"] = edge_weight
                msg_func = fn.u_mul_e("h", "_edge_weight", "m")
            graph.update_all(msg_func, fn.sum("m", "h"))
            feat = graph.ndata.pop("h")
            if edge_weight is None:
                feat = feat * norm
            # scale
            feat = feat * (1 - self.alpha)

            # initial residual connection to the first layer
            feat_0 = feat_0[: feat.size(0)] * self.alpha

            if self._project_initial_features:
                rst = feat.add_(feat_0)
                rst = th.addmm(
                    feat, feat, self.weight1, beta=(1 - self.beta), alpha=self.beta
                )
            else:
                rst = th.addmm(
                    feat, feat, self.weight1, beta=(1 - self.beta), alpha=self.beta
                )
                rst += th.addmm(
                    feat_0, feat_0, self.weight2, beta=(1 - self.beta), alpha=self.beta
                )

            if self._bias is not None:
                rst = rst + self._bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst
  

class SequentialII(torch.nn.Module):
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
        self.fc_in = torch.nn.Linear(in_features, hidden_features)
        self.layers = torch.nn.ModuleList([
            layer(
                hidden_features,
                layer=idx+1,
                **kwargs
            )
            for idx in range(depth)
        ])
        self.fc_out = torch.nn.Linear(hidden_features, out_features)
        self.activation = activation

    def forward(
            self,
            g: DGLGraph,
            h: torch.Tensor,
            **kwargs,
    ):
        """Forward pass."""
        g = g.local_var()
        h = self.fc_in(h)
        h0 = h
        for idx, layer in enumerate(self.layers):
            h = layer(g, h, h0, **kwargs)
        h = self.activation(h)
        h = self.fc_out(h)
        return h
  