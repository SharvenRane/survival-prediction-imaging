"""Synthetic survival imaging dataset where appearance encodes risk."""

import numpy as np
import torch


def make_synthetic_survival_images(
    n_samples: int = 256,
    image_size: int = 32,
    censor_fraction: float = 0.3,
    noise: float = 0.15,
    seed: int = 0,
):
    """Build a synthetic image based survival dataset.

    Each sample is a single channel image. A latent risk score drives the
    image appearance: higher risk samples are brighter (a larger constant
    offset) and contain a brighter central blob. Survival time is drawn from an
    exponential distribution whose hazard grows with the latent risk, so a
    model that reads brightness can recover risk and produce a concordance
    index well above chance.

    Args:
        n_samples: number of images to generate.
        image_size: height and width of each square image.
        censor_fraction: fraction of subjects that are right censored.
        noise: standard deviation of additive pixel noise.
        seed: random seed for reproducibility.

    Returns:
        images: float tensor of shape (N, 1, image_size, image_size) in roughly
            [0, 1] range.
        time: float tensor of shape (N,) with survival or censoring times.
        event: float tensor of shape (N,) with 1 for observed events.
        latent_risk: float tensor of shape (N,) with the ground truth risk used
            to generate the data, useful for sanity checks.
    """
    rng = np.random.default_rng(seed)

    # Latent risk per subject, standardized to zero mean unit variance.
    latent_risk = rng.normal(0.0, 1.0, size=n_samples).astype(np.float32)

    # Build images: a base brightness proportional to risk plus a central blob
    # whose intensity also scales with risk.
    images = np.zeros((n_samples, 1, image_size, image_size), dtype=np.float32)

    yy, xx = np.mgrid[0:image_size, 0:image_size]
    center = (image_size - 1) / 2.0
    radius2 = ((xx - center) ** 2 + (yy - center) ** 2).astype(np.float32)
    blob = np.exp(-radius2 / (2.0 * (image_size / 6.0) ** 2)).astype(np.float32)

    # Map latent risk to a positive brightness factor in roughly [0, 1].
    brightness = 1.0 / (1.0 + np.exp(-latent_risk))  # sigmoid

    for i in range(n_samples):
        base = 0.2 * brightness[i]
        img = base + 0.6 * brightness[i] * blob
        img = img + rng.normal(0.0, noise, size=(image_size, image_size)).astype(np.float32)
        images[i, 0] = img

    images = np.clip(images, 0.0, 1.0)

    # Survival times: exponential with hazard increasing in latent risk.
    # Higher risk gives shorter times.
    hazard = np.exp(0.9 * latent_risk).astype(np.float64)
    raw_time = rng.exponential(scale=1.0 / hazard).astype(np.float64)

    # Apply right censoring to a random subset by drawing an independent
    # censoring time and taking the minimum.
    censor_time = rng.exponential(scale=np.quantile(raw_time, 1.0 - censor_fraction) + 1e-6, size=n_samples)
    observed_time = np.minimum(raw_time, censor_time)
    event = (raw_time <= censor_time).astype(np.float32)

    images_t = torch.from_numpy(images)
    time_t = torch.from_numpy(observed_time.astype(np.float32))
    event_t = torch.from_numpy(event)
    latent_t = torch.from_numpy(latent_risk)

    return images_t, time_t, event_t, latent_t
