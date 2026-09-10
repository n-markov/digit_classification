# Digit Classification

A neural network built from scratch in NumPy, working through the fundamentals — dense layers, activations, loss, and backpropagation — before applying them to digit classification.

No deep learning framework, and no dependency beyond NumPy, Matplotlib and JupyterLab. Everything the network does is written out by hand.

## Structure

Read the notebooks in order.

- `notebooks/01_network_from_scratch.ipynb` — the derivation; builds the network up from nothing:
  - single neuron → single layer → batched examples → multi-layer network
  - `LayerDense`, `ActivationReLU`, `ActivationSoftmax`, `LossCategoricalCrossentropy`
  - manual backpropagation, plus a combined `ActivationSoftmaxLossCategoricalCrossentropy` for a faster backward step
  - `OptimizerSGD` and a training loop over a synthetic spiral dataset
- `notebooks/02_mnist_classification.ipynb` — the real thing: MNIST digits, built on the two modules below

Supporting modules, kept at the root so both notebooks can import them:

- `neuralnet.py` — the classes from notebook 01, lifted out unchanged rather than redefined
- `data/mnist.py` — MNIST loader, two interchangeable sources, cached under `data/raw/`

## Data

MNIST: 60,000 training and 10,000 test images, 28×28 greyscale, labelled 0–9.

`load_mnist()` defaults to the four original idx archives from the S3 mirror and decodes
the binary format by hand — the dimension count is read straight out of the magic number,
so images and labels share one code path. `load_mnist(source="npz")` instead pulls a single
pre-packed bundle and lets `np.load` do the work.

Both land in `data/raw/` on first call and are local from then on. Running the module
directly fetches both and asserts they are byte-identical:

```bash
uv run python data/mnist.py
```

Other routes were measured and rejected. `kagglehub` does work anonymously, with no API
key or `kaggle.json`, but it costs five transitive dependencies and a 105 MB extraction
cache to arrive at the same idx files needing the same parser. The Hugging Face parquet
copy stores each image as encoded PNG bytes, so it needs pyarrow to read the table and
Pillow to decode 70,000 rows. OpenML means adding scikit-learn, which rather defeats the
point of the exercise.

| source | transfer | on disk | extra deps | parsing to write |
| --- | --- | --- | --- | --- |
| idx mirror (default) | 9.9 MB | 55 MB | none | ~6 lines |
| npz bundle | 11.5 MB | 11.5 MB | none | none |
| kagglehub | 22 MB | 105 MB | 5 | ~6 lines |
| Hugging Face parquet | 15.6 MB | 16 MB | 2 | PNG decode per row |

## Results

A 784 to 128 to 10 network, ReLU and softmax, trained for 20 epochs on 55,000 images
in batches of 128.

| | |
| --- | --- |
| test accuracy | 97.73% |
| test loss | 0.0782 |
| errors | 227 of 10,000 |
| learning rate | 0.02, momentum 0.9, no decay |
| training time | roughly 4 minutes on CPU |

Both sanity checks pass before training starts: the network memorises 100 examples
completely, and every probed gradient agrees with a finite-difference estimate to
within 3.5e-07.

Two findings from the hyperparameter sweep are baked into the notebook's commentary.
Learning rates at or above 0.1 diverge to chance, because momentum at 0.9 multiplies
the effective step roughly tenfold. And decay, once scaled to the fact that it applies
per update rather than per epoch, made no difference worth having over a run this
short, so it ships implemented but switched off.

The most common confusion by a wide margin is a 9 read as a 4, at 25 of the 227
errors. Sorting the mistakes by confidence surfaces mostly ambiguous handwriting.

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
- [x] Real MNIST data downloaded, verified and preprocessed
- [x] He initialisation, as a `LayerDense(..., init="he")` option
- [x] Sanity checks: overfits 100 examples, backprop agrees with finite differences
- [x] Mini-batching with a per-epoch shuffle
- [x] Learning rate decay and momentum, as `OptimizerSGD` options
- [x] Trained and evaluated on MNIST: **97.7% test accuracy**
- [x] Confusion matrix and worst misclassifications
- [ ] L2 regularisation and dropout
- [ ] Adam, for comparison against momentum
- [ ] A convolutional layer
