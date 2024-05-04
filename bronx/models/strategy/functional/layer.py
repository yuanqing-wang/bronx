import torch
import gpytorch
import dgl

class GPLayer(gpytorch.models.ApproximateGP):
    def __init__(
            self,
            num_dim,
            num_tasks,
            grid_bounds=(-1, 1),
            grid_size=64,
    ):
        variational_distribution = gpytorch.variational\
        .CholeskyVariationalDistribution(
            num_inducing_points=grid_size,
            batch_shape=torch.Size([num_dim]),
        )

        # variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
        #     gpytorch.variational.GridInterpolationVariationalStrategy(
        #         self,
        #         grid_size=grid_size,
        #         grid_bounds=[grid_bounds],
        #         variational_distribution=variational_distribution,
        #     ),
        #     num_tasks=num_tasks,
        # )

        variational_strategy = gpytorch.variational.GridInterpolationVariationalStrategy(
            self,
            grid_size=grid_size,
            grid_bounds=[grid_bounds],
            variational_distribution=variational_distribution,
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
        print(x.shape, mean_x.shape, covar_x.shape)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    