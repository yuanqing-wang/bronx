import torch
import gpytorch
import dgl

class GPLayer(gpytorch.models.ApproximateGP):
    def __init__(
            self,
            num_dim,
            grid_bounds=(-1, 1),
            grid_size=64,
    ):
        variational_distribution = gpytorch.variational\
        .CholeskyVariationalDistribution(
            num_inducing_points=grid_size,
            batch_shape=torch.Size([num_dim]),
        )

        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.GridInterpolationVariationalStrategy(
                self,
                grid_size=grid_size,
                grid_bounds=[grid_bounds],
                variational_distribution=variational_distribution,
            ),
            num_tasks=num_dim,
        )

        super().__init__(variational_strategy)

        self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    ard_num_dims=num_dim,
            ),
        )

        self.mean_module = gpytorch.means.ConstantMean()
        self.grid_bounds = grid_bounds

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

class GraphGPLayer(gpytorch.Module):
    """ A layer that applys GP on graph data.

    Parameters
    ----------
    layer : torch.nn.Module
        The layer that is applied before GP.

    in_features : int
        The number of input features.

    out_features : int
        The number of output features.

    grid_bounds : tuple of float, optional
        The bounds of the grid.

    grid_size : int, optional
        The size of the grid.

    Examples
    --------
    >>> import torch
    >>> from bronx.models.zoo.dgl import GCN
    >>> layer = GraphGPLayer(
    ...     layer=GCN,
    ...     in_features=10,
    ...     out_features=20,
    ... )
    >>> g = dgl.rand_graph(5, 20)
    >>> h = torch.rand(5, 10)
    >>> h = layer(g, h)
    >>> h.shape
    torch.Size([5, 20])
    """
    def __init__(
            self,
            layer,
            in_features,
            out_features,
            grid_bounds=(-1, 1),
            grid_size=64,
    ):
        super().__init__()
        self.layer = layer(
            in_features, out_features,
        )

        self.gp_layer = GPLayer(
            num_dim=out_features,
            grid_bounds=grid_bounds,
            grid_size=grid_size,
        )

    def forward(
            self,
            g: dgl.DGLGraph,
            h: torch.Tensor,
    ):
        h = self.layer(g, h).tanh()
        h = self.gp_layer(h).rsample()
        return h


