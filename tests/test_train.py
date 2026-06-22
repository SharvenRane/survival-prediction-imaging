"""End to end behavior tests: loss decreases and c-index beats chance."""

import numpy as np

from src.data import make_synthetic_survival_images
from src.train import train_survival_model, evaluate_cindex


def test_loss_decreases_during_training():
    images, time, event, _ = make_synthetic_survival_images(
        n_samples=160, image_size=16, seed=0
    )
    _, history = train_survival_model(
        images, time, event, epochs=50, lr=1e-2, seed=0
    )
    # Compare the average of the first few epochs to the last few, which is
    # robust to small step to step noise.
    early = np.mean(history[:5])
    late = np.mean(history[-5:])
    assert late < early
    assert np.isfinite(history).all()


def test_cindex_above_chance_when_risk_encoded():
    # Train on a cohort whose images encode risk, evaluate on a held out
    # cohort drawn from the same generator. A model reading brightness should
    # rank survival clearly above the 0.5 chance line.
    train_imgs, train_t, train_e, _ = make_synthetic_survival_images(
        n_samples=240, image_size=16, noise=0.1, seed=10
    )
    test_imgs, test_t, test_e, _ = make_synthetic_survival_images(
        n_samples=240, image_size=16, noise=0.1, seed=11
    )
    model, _ = train_survival_model(
        train_imgs, train_t, train_e, epochs=80, lr=1e-2, seed=0
    )
    cindex = evaluate_cindex(model, test_imgs, test_t, test_e)
    assert cindex > 0.62
