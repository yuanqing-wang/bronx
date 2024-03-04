import torch
import pyro


def init_log_sigma(model, value):
    """Initializes the log_sigma parameters of a model
    Parameters
    ----------
    model : torch.nn.Module
        The model to initialize
    value : float
        The value to initialize the log_sigma parameters to
    """
    log_sigma_params = {
        name + "_log_sigma": pyro.nn.PyroParam(
            torch.ones(param.shape) * value,
        )
        for name, param in model.named_parameters()
    }

    for name, param in log_sigma_params.items():
        setattr(model, name, param)