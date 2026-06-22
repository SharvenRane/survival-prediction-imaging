# survival-prediction-imaging

Image based survival prediction. A small convolutional network reads an image
and produces a single log risk score, and that score feeds a Cox proportional
hazards head trained by maximizing the partial likelihood. Model quality is
measured with Harrell's concordance index, the standard rank metric for right
censored survival data.

The point of this repo is to show the full pipeline working end to end on data
where the answer is known. The synthetic dataset bakes risk directly into the
pixels, so a correct model has to recover it, and the concordance index has to
climb well past the 0.5 chance line.

## Idea

In the Cox model every subject has a hazard that scales by exp(risk), where the
risk comes from the image. Training compares, for each subject that actually
experienced an event, its risk against the risk of everyone still at risk at
that moment. You never need the absolute survival curve to fit this, only the
ordering of risks, which is why the network output is an unbounded scalar.

Two facts make the toy problem honest:

1. Higher latent risk makes an image brighter and gives it a brighter central
   blob.
2. Higher latent risk shortens survival time through an exponential hazard.

So brightness is the readable signal, and a model that learns to read it ranks
survival correctly.

## Layout

```
src/
  data.py      synthetic image survival generator
  model.py     SurvivalCNN, a compact CNN with a linear risk head
  losses.py    Cox partial likelihood (negative log, Breslow form)
  metrics.py   Harrell concordance index with censoring
  train.py     full batch training loop and c-index evaluation
tests/
  test_data.py     shapes, ranges, risk encoding sanity checks
  test_losses.py   scalar, finite, differentiable, ordering behavior
  test_metrics.py  perfect, worst, tied, censored cases
  test_train.py    loss goes down, c-index beats chance
```

## Why full batch training

The partial likelihood needs a well defined risk set: for a given event, who
else was still at risk. A complete pass over the cohort keeps that set intact
and gives a stable gradient on small data. On large cohorts you would move to
mini batches with a within batch risk set or a sampled approximation, but for a
few hundred synthetic images full batch is both correct and fast on CPU.

## Numerical detail

The loss sorts subjects by descending time and uses `torch.logcumsumexp` to get,
for each subject, the log sum of exp(risk) over its risk set in one stable pass.
That avoids overflow from exponentiating raw risk scores.

## Running the tests

```
python -m pytest tests/ -q
```

Everything runs on CPU with tiny tensors and synthetic data, so there is no
download and no GPU requirement. On a recent run all 15 tests passed in about
two seconds. The end to end test trains on one synthetic cohort and evaluates
the concordance index on a separate held out cohort drawn from the same
generator, and it asserts the index lands clearly above chance.

## Using it on your own data

Replace `make_synthetic_survival_images` with a loader that yields an image
tensor of shape (N, C, H, W), a time tensor (N,), and an event tensor (N,) where
1 marks an observed event and 0 marks right censoring. The model, loss, and
metric do not care where the images came from.
