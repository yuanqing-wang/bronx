import torch

class EdgeModel(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
    ):
        super().__init__()
        self.fc = torch.nn.Linear(in_features, out_features)

    