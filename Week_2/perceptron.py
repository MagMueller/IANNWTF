from numpy.lib.function_base import select
from preparation import sigmoid, sigmoidprime
import numpy as np


class Perceptron():
    """One single Perceptron with weights and alpha which can be initilized in the beginning. """

    def __init__(self, input_units, alpha=1) -> None:
        self.alpha = alpha
        self.weights = np.random.normal(size=input_units + 1)

    def forward_step(self, inputs):
        self.raw_inputs = np.concatenate(
            (inputs, np.array([1])))
        self.sum_inputs = np.sum(self.raw_inputs @ self.weights)
        out = sigmoid(self.sum_inputs)
        self.out = out
        return self.out

    def update(self, delta):
        """Updates the weights."""
        #gradients = delta
        self.weights -= self.alpha * delta * self.raw_inputs

    def get_output(self):
        """Get the output after the activation function."""
        return self.out

    def get_input(self):
        """Get the input to the perceptron before the activation function."""
        return self.sum_inputs

    def get_weights(self):
        return self.weights


p = Perceptron(2)
o = p.forward_step(np.array([4, 5]))
