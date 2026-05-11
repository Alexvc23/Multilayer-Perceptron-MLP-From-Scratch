"""
Network Orchestrator

This module defines the MultilayerPerceptron class that strings together
multiple Dense layers, managing the execution of both the forward and backward passes.
"""
import numpy as np

class MultilayerPerceptron:
    """
    Main Neural Network class that coordinates layers, calculates the full forward pass,
    and iterates weight updates via backpropagation.
    """
    
    def __init__(self):
        self.layers = []
        
    def add_layer(self, layer):
        """Append a layer to the network."""
        pass
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Executes the feedforward pass sequentially through all layers.
        """
        pass
        
    def backward(self, loss_grad: np.ndarray, learning_rate: float):
        """
        Executes the backpropagation pass to update weights and biases based on the error.
        """
        pass
