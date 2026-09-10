"""Neural network building blocks, lifted verbatim from `notebooks/01_network_from_scratch.ipynb`.

That notebook stays as the derivation; this module is the reusable version so the
MNIST notebook can stay about digits.
"""

from abc import ABC, abstractmethod

import numpy as np


class LayerDense:
    """A fully connected layer.

    `init="fixed"` is the flat 0.1 scaling used throughout notebook 01, kept as the
    default so that notebook still reproduces. `init="he"` scales by the fan-in
    instead, which is what keeps pre-activations from blowing up once the input is
    784 pixels wide rather than 2 spiral coordinates.
    """

    def __init__(self, n_inputs, n_neurons, init="fixed", dtype=np.float64) -> None:
        scales = {
            "fixed": 0.1,
            "he": np.sqrt(2.0 / n_inputs),
        }

        if init not in scales:
            raise ValueError(f"unknown init {init!r}, expected one of {sorted(scales)}")

        # Pre transposed
        self.weights = (scales[init] * np.random.randn(n_inputs, n_neurons)).astype(dtype)
        self.biases = np.zeros((1, n_neurons), dtype=dtype)

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class ActivationReLU:
    def __init__(self) -> None:
        pass

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class ActivationSoftmax:
    def __init__(self) -> None:
        ...

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probs

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class Loss(ABC):
    def __init__(self) -> None:
        ...

    @abstractmethod
    def forward(self, output, y):
        pass

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class LossCategoricalCrossentropy(Loss):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples


class ActivationSoftmaxLossCategoricalCrossentropy:
    def __init__(self) -> None:
        self.activation = ActivationSoftmax()
        self.loss = LossCategoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples


class OptimizerSGD:
    """Stochastic gradient descent, optionally with decay and momentum.

    Both extras default to off, so calling this with a learning rate alone behaves
    exactly as it did in notebook 01. With them on, wrap each step:

        optimizer.pre_update_params()
        optimizer.update_params(layer)   # once per layer
        optimizer.post_update_params()

    Decay shrinks the learning rate as 1 / (1 + decay * iterations), taking large
    steps early and fine ones later. Momentum carries a fraction of the previous
    update into the current one, which damps the zig-zag across narrow valleys and
    builds speed along directions the gradient keeps agreeing on.
    """

    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0) -> None:
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.iterations = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1 + self.decay * self.iterations)

    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = (
                self.momentum * layer.weight_momentums
                - self.current_learning_rate * layer.dweights
            )
            bias_updates = (
                self.momentum * layer.bias_momentums
                - self.current_learning_rate * layer.dbiases
            )

            layer.weight_momentums = weight_updates
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1
