from typing import ParamSpec
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
    params = {}

    for name, base_name in model.buffered_params.items():
        mean_name = base_name + "_mean"
        sigma_name = base_name + "_sigma"
        mean = getattr(model, mean_name)
        sigma = getattr(model, sigma_name)
        params[name] = pyro.nn.PyroSample(
            pyro.distributions.Normal(
                mean,sigma
            ).to_event(mean.dim())
        )

    for name, param in params.items():
        rsetattr(model, name, params[name])