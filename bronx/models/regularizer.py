import torch
import pyro

class ConsistencyRegularizer(torch.nn.Module):
    def __init__(self, temperature, factor):
        super().__init__()
        self.temperature = temperature
        self.factor = factor

    def forward(self, probs):
        if probs.dim() == 2:
            return 0.0
        avg_probs = probs
        while avg_probs.dim() > 2:
            avg_probs = avg_probs.mean(0)
        sharpened_probs = avg_probs.pow(1.0 / self.temperature)
        _shapened_probs = sharpened_probs
        sharpened_probs = sharpened_probs / (sharpened_probs.sum(-1, keepdims=True) + 1e-5)
        loss = (sharpened_probs - probs).pow(2).mean()

        # f.write(str(sharpened_probs) + "\n")
        #     f.write(str(probs) + "\n")
        #     f.write(f"ConsistencyRegularizer: {loss}, factor: {self.factor}\n")

        # print(probs.shape)
        pyro.factor("consistency_regularizer", -loss * self.factor)
        return loss