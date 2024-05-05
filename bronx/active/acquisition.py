import torch
import pyro

def expected_improvement(
        model: torch.nn.Module,
        g: torch.Tensor,
        h: torch.Tensor,
        best: torch.Tensor = 0.0,
        num_samples: int = 1000,
):
    predictive = pyro.infer.Predictive(
        model.svi.model,
        guide=model.svi.guide,
        num_samples=num_samples,
        parallel=False,
        return_sites=["_RETURN", "y"],
    )

    y_hat = predictive(g, h, y=None)["y"]
    ei = (y_hat - best).clamp(min=0).mean(dim=0)
    return ei

def probability_of_improvement(
        model: torch.nn.Module,
        g: torch.Tensor,
        h: torch.Tensor,
        best: torch.Tensor = 0.0,
        num_samples: int = 1000,
):
    predictive = pyro.infer.Predictive(
        model.svi.model,
        guide=model.svi.guide,
        num_samples=num_samples,
        parallel=False,
        return_sites=["_RETURN", "y"],
    )

    y_hat = predictive(g, h, y=None)["y"]
    pi = (y_hat > best).float().mean(dim=0)
    return pi


    