import abc
import torch

class BronxModel(abc.ABC):
    """Base class for all models in the Bronx framework.

    Methods
    -------
    __call__(self, *args, **kwargs)
        Make a prediction using the model.

    train(self, *args, **kwargs)
        Train the model using the given data.
    """
    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> torch.distributions.Distribution:
        pass

    @abc.abstractmethod
    def train(self, *args, **kwargs) -> None:
        pass

