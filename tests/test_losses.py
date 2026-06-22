"""Behavior tests for the Cox partial likelihood loss."""

import torch

from src.losses import cox_ph_loss


def test_loss_is_scalar_and_finite():
    torch.manual_seed(0)
    n = 20
    risk = torch.randn(n)
    time = torch.rand(n) + 0.1
    event = (torch.rand(n) > 0.4).float()
    loss = cox_ph_loss(risk, time, event)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_correct_ordering_beats_wrong_ordering():
    # Two subjects, both events, subject 0 fails first so it should carry the
    # higher risk for a good model. Higher risk on the earlier failure must
    # give a smaller negative log partial likelihood.
    time = torch.tensor([1.0, 2.0])
    event = torch.tensor([1.0, 1.0])

    good_risk = torch.tensor([2.0, 0.0])  # earlier failure has higher risk
    bad_risk = torch.tensor([0.0, 2.0])   # earlier failure has lower risk

    good_loss = cox_ph_loss(good_risk, time, event)
    bad_loss = cox_ph_loss(bad_risk, time, event)
    assert good_loss < bad_loss


def test_no_events_returns_zero():
    risk = torch.randn(8, requires_grad=True)
    time = torch.rand(8) + 0.1
    event = torch.zeros(8)
    loss = cox_ph_loss(risk, time, event)
    assert float(loss) == 0.0


def test_loss_is_differentiable():
    torch.manual_seed(1)
    risk = torch.randn(16, requires_grad=True)
    time = torch.rand(16) + 0.1
    event = (torch.rand(16) > 0.3).float()
    loss = cox_ph_loss(risk, time, event)
    loss.backward()
    assert risk.grad is not None
    assert torch.isfinite(risk.grad).all()
