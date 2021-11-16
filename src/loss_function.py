import torch
import torch.nn as nn


def mse(yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    loss = nn.MSELoss()
    return loss(yhat, y)


def rmse(yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    loss = RMSELoss()
    return loss(yhat, y)


def identity(x: torch.Tensor) -> torch.Tensor:
    return x.clone().detach()


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss
