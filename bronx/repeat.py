import torch
import lightning as pl

class Repeat(pl.LightningModule):
    def __init__(
            models,
    ):
        super().__init__()
        