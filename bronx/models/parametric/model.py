import torch
from .utils import init_log_sigma
from ..model import BronxModel

class ParametricModel(BronxModel):
    def __init__(
            self,
            model: torch.nn.Module,
            log_sigma: float = 0.0,
    ):
        self.model = model
        init_log_sigma(self.model, log_sigma)

    def forward(self, x):
        return self.model(x)
