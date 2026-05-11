"""
Activation Functions

Contains mathematical implementations for non-linear activations and their derivatives,
essential for computing the chain rule during backpropagation.
"""
import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Applies the sigmoid function to squash values into the range (0, 1).
    """
    pass

def sigmoid_prime(x: np.ndarray) -> np.ndarray:
    """
    Derivative of the sigmoid function, used during the backward pass.
    """
    pass

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Applies the softmax function for output probability distribution.
    """
    pass
