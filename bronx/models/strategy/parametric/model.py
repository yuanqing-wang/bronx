import torch
import pyro
import dgl
from .utils import init_sigma, to_pyro_module_
from ...zoo.dgl import Sequential
from ...model import BronxLightningWrapper, BronxModel, BronxPyroMixin
from ....global_parameters import NUM_SAMPLES

class UnwrappedParametricModel(pyro.nn.PyroModule):
    """ A model that characterizes the parametric uncertainty of a graph.
    
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

    activation : torch.nn.Module
        The activation function.

    proj_in : bool
        Whether to project the input features.

    proj_out : bool
        Whether to project the output features.
    """

    def __init__(
            self,
            layer: torch.nn.Module,
            in_features: int,
            out_features: int,
            hidden_features: int,
            depth: int,
            activation: torch.nn.Module = torch.nn.SiLU(),
            proj_in: bool = False,
            proj_out: bool = False,
            sigma: float = 1.0,
            *args, **kwargs,
    ):
        super().__init__()
        if proj_in:
            self.proj_in = torch.nn.Linear(in_features, hidden_features)
        if proj_out:
            self.proj_out = torch.nn.Linear(hidden_features, out_features)
        
        self.layers = layer.sequential()(
            layer=layer,
            depth=depth,
            activation=activation,
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
        )

        self.sigma = sigma
        self._prepare_buffer()

    def _prepare_buffer(self):
        buffered_params = {}
        for name, param in self.named_parameters():
            base_name = name.replace(".", "-")
            mean_name = base_name + "_mean"
            sigma_name = base_name + "_sigma"
            self.register_buffer(mean_name, torch.zeros(param.shape))
            self.register_buffer(sigma_name, torch.ones(param.shape) * self.sigma)
            buffered_params[name] = base_name
        self.buffered_params = buffered_params

    def forward(
            self,
            g: dgl.DGLGraph,
            h: torch.Tensor,
    ):
        if hasattr(self, "proj_in"):
            h = self.proj_in(h)
        h = self.layers(g, h)
        if hasattr(self, "proj_out"):
            h = self.proj_out(h)
        return h 
    
class ParametricModel(BronxPyroMixin, BronxLightningWrapper):
    def __init__(
            self, 
            autoguide: pyro.infer.autoguide.guides.AutoGuide,
            head: str,
            sigma: float = 1.0,
            optimizer: str = "Adam",
            lr: float = 1e-2,
            weight_decay: float = 1e-3,
            loss: torch.nn.Module = pyro.infer.Trace_ELBO(),
            *args, 
            **kwargs,
    ):
        model = UnwrappedParametricModel(*args, **kwargs)
        to_pyro_module_(model)
        init_sigma(model, sigma)
        super().__init__(model)

        self.model.guide = autoguide(model)

        # initialize head
        self.head = head()
        self.automatic_optimization = False
        self.save_hyperparameters()

        # initialize optimizer
        optimizer = getattr(pyro.optim, optimizer)(
            {"lr": lr, "weight_decay": weight_decay},
        )

        self.svi = pyro.infer.SVI(
            self.forward,
            self.guide,
            optim=optimizer,
            loss=loss,
            num_samples=NUM_SAMPLES,
        )

    def configure_optimizers(self):
        return None

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        init_sigma(self.model, self.model.sigma)
        return super().to(*args, **kwargs)

    def cuda(self, *args, **kwargs):
        self.model.cuda(*args, **kwargs)
        init_sigma(self.model, self.model.sigma)
        return super().cuda(*args, **kwargs)
    
    def cpu(self, *args, **kwargs):
        self.model.cpu(*args, **kwargs)
        init_sigma(self.model, self.model.sigma)
        return super().cpu(*args, **kwargs)
    

# =============================================================================
# per node models
# =============================================================================
from .layers.per_node import GCN
class UnwrappedPerNodeParametricModel(pyro.nn.PyroModule):
    """ A model that characterizes the parametric uncertainty of a graph.
    
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

    activation : torch.nn.Module
        The activation function.

    proj_in : bool
        Whether to project the input features.

    proj_out : bool
        Whether to project the output features.
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            hidden_features: int,
            depth: int,
            activation: torch.nn.Module = torch.nn.SiLU(),
            proj_in: bool = False,
            proj_out: bool = False,
            *args, **kwargs,
    ):
        super().__init__()
        if proj_in:
            self.proj_in = torch.nn.Linear(in_features, hidden_features)
        if proj_out:
            self.proj_out = torch.nn.Linear(hidden_features, out_features)

        self.activation = activation
        self.layers = torch.nn.ModuleList([
            GCN(
                in_features=in_features if idx == 0 and ~proj_in else hidden_features,
                out_features=out_features if idx == depth - 1 and ~proj_out else hidden_features,
                suffix=str(idx),
            ) for idx in range(depth)
        ])
    
    def forward(
            self,
            g: dgl.DGLGraph,
            h: torch.Tensor,
    ):
        if hasattr(self, "proj_in"):
            h = self.proj_in(h)
        for layer in self.layers:
            h = layer.forward(g, h)
            h = self.activation(h)
        if hasattr(self, "proj_out"):
            h = self.proj_out(h)
        return h 

    def guide(
            self,
            g: dgl.DGLGraph,
            h: torch.Tensor,
    ):
        if hasattr(self, "proj_in"):
            h = self.proj_in(h)
        for layer in self.layers:
            h = layer.guide(g, h)
            h = self.activation(h)
        if hasattr(self, "proj_out"):
            h = self.proj_out(h)
        return h

class PerNodeParametricModel(BronxPyroMixin, BronxLightningWrapper):
    def __init__(
            self, 
            head: str,
            sigma: float = 1.0,
            optimizer: str = "Adam",
            lr: float = 1e-2,
            weight_decay: float = 1e-3,
            loss: torch.nn.Module = pyro.infer.Trace_ELBO(),
            *args, 
            **kwargs,
    ):
        model = UnwrappedPerNodeParametricModel(*args, **kwargs)
        super().__init__(model)

        # initialize head
        self.head = head()
        self.automatic_optimization = False
        self.save_hyperparameters()

        # initialize optimizer
        optimizer = getattr(pyro.optim, optimizer)(
            {"lr": lr, "weight_decay": weight_decay},
        )

        self.svi = pyro.infer.SVI(
            self.forward,
            self.guide,
            optim=optimizer,
            loss=loss,
            num_samples=NUM_SAMPLES,
        )

    def configure_optimizers(self):
        return None
