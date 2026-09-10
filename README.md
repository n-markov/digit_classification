# Digit Classification

A neural network built from scratch in NumPy, working through the fundamentals — dense layers, activations, loss, and backpropagation — before applying them to digit classification.

## Structure

- `trial_jupyter.ipynb` — main working notebook; builds up a neural network from scratch:
  - single neuron → single layer → batched examples → multi-layer network
  - `LayerDense`, `ActivationReLU`, `ActivationSoftmax`, `LossCategoricalCrossentropy`
  - manual backpropagation, plus a combined `ActivationSoftmaxLossCategoricalCrossentropy` for a faster backward step
  - `OptimizerSGD` and a training loop over the spiral dataset
- `trial.py` / `main.py` — early scratch scripts, superseded by the notebook
- `data/spiral_data.py` — synthetic spiral dataset generator, used to test the network before moving to real digit data

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
uv run jupyter lab
```

## Progress

- [x] Single neuron / single layer forward pass
- [x] Batched inputs across multiple examples
- [x] Multi-layer dense network with ReLU and Softmax activations
- [x] Categorical cross-entropy loss
- [x] Backpropagation (manual and combined Softmax/CCE)
- [x] Training loop / optimizer (plain SGD)
- [ ] Learning rate decay / momentum
- [ ] Apply to real digit classification data
