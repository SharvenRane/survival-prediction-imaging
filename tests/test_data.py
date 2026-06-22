"""Tests for the synthetic survival imaging dataset."""

import numpy as np
import torch

from src.data import make_synthetic_survival_images


def test_shapes_and_ranges():
    images, time, event, latent = make_synthetic_survival_images(
        n_samples=64, image_size=16, seed=0
    )
    assert images.shape == (64, 1, 16, 16)
    assert time.shape == (64,)
    assert event.shape == (64,)
    assert latent.shape == (64,)
    assert float(images.min()) >= 0.0
    assert float(images.max()) <= 1.0
    assert torch.all(time > 0)


def test_some_events_and_some_censoring():
    _, _, event, _ = make_synthetic_survival_images(
        n_samples=200, censor_fraction=0.3, seed=1
    )
    e = event.numpy()
    assert e.sum() > 0           # at least some observed events
    assert (1 - e).sum() > 0     # at least some censored


def test_risk_encoded_in_brightness():
    # Higher latent risk should give brighter images on average, which is the
    # signal the CNN learns to read.
    images, _, _, latent = make_synthetic_survival_images(
        n_samples=300, image_size=16, noise=0.05, seed=2
    )
    mean_brightness = images.view(300, -1).mean(dim=1).numpy()
    corr = np.corrcoef(mean_brightness, latent.numpy())[0, 1]
    assert corr > 0.5


def test_higher_risk_shorter_time():
    # The data generator ties higher latent risk to shorter survival.
    _, time, event, latent = make_synthetic_survival_images(
        n_samples=400, seed=3
    )
    observed = event.numpy() == 1
    corr = np.corrcoef(latent.numpy()[observed], time.numpy()[observed])[0, 1]
    assert corr < 0.0
