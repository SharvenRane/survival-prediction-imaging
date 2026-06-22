"""Cox proportional hazards partial likelihood loss."""

import torch


def cox_ph_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """Negative log Cox partial likelihood.

    The model outputs a scalar log risk score per sample. Under the Cox
    proportional hazards model the partial likelihood compares, for every
    subject that experiences an event, its risk against the risk of every
    subject still at risk at that subject's event time (the risk set).

    We use the standard Breslow form. For numerical stability we compute the
    log cumulative sum of exponentiated risks over subjects sorted by
    descending time, which gives, for each subject, the log sum of exp(risk)
    over all subjects with an equal or later time (its risk set).

    Args:
        risk: shape (N,) tensor of log risk scores.
        time: shape (N,) tensor of survival or censoring times.
        event: shape (N,) tensor, 1 if the event was observed, 0 if censored.

    Returns:
        Scalar tensor with the mean negative log partial likelihood over the
        observed events.
    """
    risk = risk.view(-1)
    time = time.view(-1).to(risk.dtype)
    event = event.view(-1).to(risk.dtype)

    # Sort by descending time so that for index i the subjects j >= i form the
    # risk set (everyone with time <= time[i] does not, but with descending
    # order cumulative-from-top gives time >= time[i]).
    order = torch.argsort(time, descending=True)
    risk = risk[order]
    event = event[order]

    # log sum exp of risks over the risk set for each subject, computed as a
    # numerically stable running logcumsumexp from the top of the sorted list.
    log_cumulative_risk = torch.logcumsumexp(risk, dim=0)

    # Partial likelihood contribution only for observed events.
    uncensored_likelihood = risk - log_cumulative_risk
    num_events = event.sum()
    if num_events.item() == 0:
        # No events means no defined partial likelihood; return a zero that
        # still carries a gradient path so the caller can skip without a NaN.
        return (risk.sum() * 0.0)

    neg_log_likelihood = -(uncensored_likelihood * event).sum() / num_events
    return neg_log_likelihood
