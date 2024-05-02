import torch
import pyro

class ConsistencyRegularizer(torch.nn.Module):
    def __init__(self, temperature, factor):
        super().__init__()
        self.temperature = temperature
        self.factor = factor

    def forward(self, probs):
        if probs.dim() == 2:
            avg_probs = probs
        else:
            avg_probs = probs.mean(0)

        sharpened_probs = avg_probs.pow(1.0 / self.temperature)
        sharpened_probs = sharpened_probs / sharpened_probs.sum(-1, keepdims=True)
        loss = (sharpened_probs - probs).pow(2).sum()
        pyro.factor("consistency_regularizer", -loss * self.factor)
        return loss