"""Small CNN with a Cox proportional hazards output head."""

import torch
import torch.nn as nn


class SurvivalCNN(nn.Module):
    """A compact convolutional network that outputs a scalar log risk score.

    The convolutional trunk produces a feature vector through global average
    pooling, and a single linear layer maps it to one log risk value per image.
    The output is intentionally unbounded because the Cox partial likelihood
    operates on relative risk and only differences between scores matter.
    """

    def __init__(self, in_channels: int = 1, width: int = 16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(width * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a shape (N,) tensor of log risk scores."""
        h = self.features(x)
        h = torch.flatten(h, 1)
        risk = self.head(h)
        return risk.view(-1)
