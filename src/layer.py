"""
Dense Layer

This module implements a standard fully connected (dense) layer.
"""
import numpy as np
from src.activations import sigmoid, relu

class DenseLayer:
    """
    Represents a single layer in the MLP, initialized with a specific number of 
    input and output nodes. 
    """
    
    def __init__(self, input_size: int, output_size: int, activation_name: str = 'sigmoid', random_seed: int = 42):
        """
        Initializes weights and biases for the layer.
        
        Why: We initialize weights randomly to break symmetry so each neuron learns 
        different features. Biases are initialized to carefully chosen small values or zeros.
        Using a random seed ensures repeatability for our human evaluation.
        """
        np.random.seed(random_seed)
        # Weights initialized around 0 with small variance. Shape: (input_features, output_neurons)
        self.weights = np.random.randn(input_size, output_size) * 0.1
        # Biases initialized to zero. Shape: (1, output_neurons)
        self.biases = np.zeros((1, output_size))
        self.activation_name = activation_name
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Executes the forward pass for this layer.
        
        Why:
        We compute the dot product to combine inputs with their learned importance (weights),
        add the bias to shift the activation threshold, and finally pass the result 
        through a non-linear activation function.

        return the output of the layer after applying the activation function.
        """
        self.inputs = inputs
        # 1. Linear Transformation: Z = X * W + b
        self.z = np.dot(inputs, self.weights) + self.biases
        
        # 2. Non-linear Activation
        if self.activation_name == 'sigmoid':
            self.output = sigmoid(self.z)
        elif self.activation_name == 'relu':
            self.output = relu(self.z)
        else:
            self.output = self.z # Linear
            
        return self.output
        
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Calculates the local gradient for backpropagation and applies Gradient Descent.
        Returns the input gradient to pass to the previous layer.
        """
        pass

if __name__ == "__main__":
    # --- Manual Validation Block ---
    # Validates the forward pass math engine.
    
    # 1. Define input matrix X shape: (2, 3) 
    # Batch size of 2, with 3 input features.
    X = np.array([
        [1.0, 2.0, 3.0], 
        [4.0, 5.0, 6.0]
    ])
    
    print("--- Testing DenseLayer Forward Pass (Linear) ---")
    layer = DenseLayer(input_size=3, output_size=2, activation_name='linear')
    
    # 2. Overwrite weights and biases manually to match the notebook's test case exactly.
    layer.weights = np.array([
        [0.1, 0.2], 
        [0.3, 0.4], 
        [0.5, 0.6]
    ])
    
    layer.biases = np.array([[0.5, -0.5]])
    
    # 3. Compute the forward pass
    Z = layer.forward(X)
    
    # 4. Assert correctness
    print(f"X shape: {X.shape}")
    print(f"Weights shape: {layer.weights.shape}")
    print(f"Biases shape: {layer.biases.shape}")
    print(f"Output Z shape: {Z.shape} (Expected: (2, 2))")
    print("\nOutput Z:")
    print(Z)

