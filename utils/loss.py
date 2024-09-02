import torch
import torch.nn as nn


# High-Order MSE Loss with p = 4
class HighOrderMSELoss(nn.Module):
    def __init__(self, p=4):
        super(HighOrderMSELoss, self).__init__()
        self.p = p

    def forward(self, predictions, targets):
        # Compute the high-order difference
        loss = torch.mean((predictions - targets) ** self.p)
        return loss
    
# Exponential MSE Loss
class ExponentialMSELoss(nn.Module):
    def __init__(self):
        super(ExponentialMSELoss, self).__init__()

    def forward(self, predictions, targets):
        # Compute the exponential of the squared differences
        loss = torch.mean(torch.exp((predictions - targets) ** 2))
        return loss