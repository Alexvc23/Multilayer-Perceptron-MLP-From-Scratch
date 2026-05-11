"""
Dense Layer

This module implements a standard fully connected (dense) layer.
"""
import numpy as np

class DenseLayer:
    """
    Represents a single layer in the MLP, initialized with a specific number of 
    input and output nodes. 
    """
    
    def __init__(self, input_size: int, output_size: int, random_seed: int = 42):
        """
        Initializes weights and biases for the layer.
        """
        np.random.seed(random_seed)
        pass
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Computes the weighted sum of inputs plus bias: z = (W * x) + b
        """
        pass
        
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Calculates the local gradient for backpropagation and applies Gradient Descent.
        Returns the input gradient to pass to the previous layer.
        """
        pass
