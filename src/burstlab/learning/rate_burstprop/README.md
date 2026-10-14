AI DISCLOSURE: this file was written by Claude Opus.

# Exercise: rate-model Burstprop

> Learning material, not a project document (see `src/burstlab/AGENTS.md`). For facts, use the papers.

**Goal:** see for yourself why "burst rate with the teacher minus burst rate without it" behaves like backprop's
error signal. You'll implement Payeur et al. 2021's rate model (Methods, "Deep network model") as five small
functions and check them against PyTorch autograd. It's a short exercise. The main project then moves on to the
spiking model (Stuck 2025), which works differently in important ways (see "What doesn't carry over" below).

## Files

| File | What it is |
|---|---|
| `burstprop.py` | **Your part.** Five functions with docstrings giving the equations; the bodies raise `NotImplementedError` |
| `test_burstprop.py` | The tests that define what the functions must do, grouped in steps |
| `train_mnist.py` | Once the tests pass: trains Burstprop and backprop side by side on MNIST |

## Steps

Run the tests from the repo root. They all fail at first:

```bash
uv run pytest src/burstlab/learning/rate_burstprop -q
```

Work through `burstprop.py` top to bottom. The test file's step headers match:

1. **`forward`**: a plain sigmoid MLP that returns every layer's rates.
2. **`output_burst_probs`**: the teacher. With no error, the burst probability stays at baseline. Below target, it
   bursts more.
3. **`hidden_burst_probs`**: the dendrite. Zero feedback gives p = 0.5. Note the h(e) = 1 − e factor.
4. **`burst_deltas`**: two top-down passes (without and with the teacher). Check that zero error gives exactly zero
   deltas, everywhere.
5. **`weight_updates`**: the plasticity rule. The key tests are here:
   - At the output layer, the update is *exactly* `p_baseline ×` backprop's update.
   - At hidden layers it points the same way as backprop's (cosine > 0.99) and is about 4× smaller per layer below
     the output.

A single test can be run with e.g. `-k output`. Try writing everything before looking at
`~/dev/clones/Burstprop/layers.py`.

## Questions to work out on paper

- Why is the output-layer update exactly proportional to backprop's? (Write out −(b_t − b) for the output layer.)
- Where does the factor 1/4 per layer come from? (Hint: σ(u_t) − σ(u) ≈ σ′(u)·(u_t − u), and what is u at
  initialization?)
- When would the approximation break down? What happens as training makes the dendritic inputs larger?
- `train_mnist.py` gives each layer down 4× the learning rate. What would happen without that?

## Then run MNIST

```bash
uv run python src/burstlab/learning/rate_burstprop/train_mnist.py
```

This downloads MNIST to `data/` at the repo root; `--data-root src/unm_nmc_burstprop` reuses the copy already in
the repo. With the defaults (1×100 hidden, 5 epochs, a few seconds), Burstprop and backprop should reach about 97%
and stay within a fraction of a percent of each other. The hidden-layer angle should stay near 0°.

Things to try:

- `--hidden 500 500 500 --lr-burstprop 0.05 --lr-backprop 0.01`: a deep sigmoid network. Both methods train slowly
  with the plain rule (no momentum). That's a known property of deep sigmoid networks, not a Burstprop failure.
  Watch whether the angle grows with depth.
- Raise the learning rates until something breaks. Does Burstprop break before backprop, or with it?
- `--p-baseline`: what changes, and why does the output-layer exactness test still hold?

## What doesn't carry over to the spiking model

This rate model is *not* the expected value of Stuck's spiking network:

- Here the teacher and no-teacher passes are computed separately and subtracted, so the baseline cancels exactly.
  Stuck's network runs one online pass. The unsigned model uses a fixed baseline instead, which is why it can leak
  a baseline into hidden dendrites.
- Here f′ enters exactly through h(e). In Stuck's model it enters only through gating by postsynaptic events.
- There's no spiking noise here: every unit's error is a real number, not a few Bernoulli bursts.

So passing here shows that the core idea works. It doesn't show that the spiking version will.
