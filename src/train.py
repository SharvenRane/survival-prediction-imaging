"""Training loop for the image based Cox survival model."""

import torch

from .model import SurvivalCNN
from .losses import cox_ph_loss
from .metrics import concordance_index


def train_survival_model(
    images: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    epochs: int = 60,
    lr: float = 1e-2,
    width: int = 16,
    seed: int = 0,
):
    """Fit a SurvivalCNN with full batch gradient descent.

    Full batch training keeps the risk set well defined: the Cox partial
    likelihood compares every event against all later subjects, so a complete
    pass over the cohort gives a stable gradient on small synthetic data.

    Args:
        images: tensor (N, C, H, W).
        time: tensor (N,) of times.
        event: tensor (N,) of event indicators.
        epochs: number of full batch optimization steps.
        lr: learning rate for Adam.
        width: base channel width of the CNN.
        seed: torch manual seed.

    Returns:
        model: the trained SurvivalCNN.
        history: list of float loss values, one per epoch.
    """
    torch.manual_seed(seed)
    in_channels = images.shape[1]
    model = SurvivalCNN(in_channels=in_channels, width=width)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        risk = model(images)
        loss = cox_ph_loss(risk, time, event)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))

    return model, history


@torch.no_grad()
def evaluate_cindex(model, images, time, event) -> float:
    """Compute the concordance index of a trained model on a cohort."""
    model.eval()
    risk = model(images).cpu().numpy()
    return concordance_index(time.cpu().numpy(), event.cpu().numpy(), risk)
