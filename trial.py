import numpy as np

inputs = np.array([1.2, 5.1, 2.1])
weights = np.array([3.1, 2.1, 8.7])
bias = 3

output = np.dot(inputs, weights) + bias
print(output)

# 3 neurons with 4 inputs

np.random.seed(42)
inputs = [np.random.uniform(0, 5, size = 4) for i in range(3)]
weights = [np.random.uniform(0, 2, size = 4) for i in range(3)]
bias = [np.random.uniform(0, 1) for i in range(3)]

output = [np.dot(inputs[i], weights[i]) + bias[i] for i in range(3)]

print([f"{x:.2f}" for x in output])

# Parameterise neurons and inputs

def foo(neuron_size, input_size):
    np.random.seed(42)

    inputs = [np.random.uniform(0, 5, size = input_size) for _ in range(neuron_size)]
    weights = [np.random.uniform(0, 2, size = input_size) for _ in range(neuron_size)]
    bias = [np.random.uniform(0, 1) for i in range(neuron_size)]

    raw_output = [np.dot(inputs[i], weights[i]) + bias[i] for i in range(neuron_size)]
    output = [f"{x:.2f}" for x in raw_output]

    return output

print(foo(3, 4))