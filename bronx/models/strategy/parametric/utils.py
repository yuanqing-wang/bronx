import torch
import pyro
from pyro.nn.module import to_pyro_module_

import functools

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def init_sigma(model, value):
    """Initializes the log_sigma parameters of a model

    Parameters
    ----------
    model : torch.nn.Module
        The model to initialize

    value : float
        The value to initialize the log_sigma parameters to

    """
    params = {
        name: pyro.nn.PyroSample(
            pyro.distributions.Normal(
                torch.zeros(param.shape),
                torch.ones(param.shape) * value,
            ).to_event(param.dim())
        )
        for name, param in model.named_parameters()
    }

    for name, param in params.items():
        rsetattr(model, name, params[name])