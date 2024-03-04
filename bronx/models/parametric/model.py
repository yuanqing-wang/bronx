import torch
from .utils import init_log_sigma
from ..model import BronxModel

class ParametricModel(BronxModel):
    def __init__(
            self,
            model: torch.nn.Module,
    ):
        self.model = model