"""
Loss Functions

Implements the Binary Cross-Entropy loss formula required to evaluate
classification errors on the output layer.
"""
import numpy as np

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the exact BCE formula: E = - (1/N) * sum( y*log(y_pred) + (1-y)*log(1-y_pred) )
    """
    pass

def binary_cross_entropy_prime(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculates the initial derivative (gradient) of BCE, to jumpstart backpropagation.
    """
    pass
