"""
Network Orchestrator

This module defines the MultilayerPerceptron class that strings together
multiple Dense layers, managing the execution of both the forward and backward passes.
"""
import numpy as np
import json
from src.layer import DenseLayer

class MultilayerPerceptron:
    """
    Main Neural Network class that coordinates layers, calculates the full forward pass,
    and iterates weight updates via backpropagation.
    """
    
    def __init__(self, topology: list, hidden_activation: str = 'sigmoid', output_activation: str = 'sigmoid'):
        """
        Initializes the network based on a list defining the size of each layer.
        
        Args:
            topology: A list of integers (e.g., [30, 24, 24, 1]).
            
        Why dynamically construct layers?
        This allows the human developer (you) to easily experiment with network capacity.
        Adding more neurons or layers increases the network's ability to model complex 
        non-linear boundaries, though it risks overfitting.
        """
        self.layers = []
        
        # Iterate through the topology to connect each layer i to layer i+1
        for i in range(len(topology) - 1):
            input_size = topology[i]
            output_size = topology[i + 1]
            
            # The final layer typically uses a specific activation (like Sigmoid or Softmax) 
            is_final_layer = (i == len(topology) - 2)
            activation = output_activation if is_final_layer else hidden_activation
            
            layer = DenseLayer(input_size, output_size, activation_name=activation)
            self.layers.append(layer)
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Passes the input data sequentially through all layers.
        
        Returns:
            The final prediction probabilities of the network.
        """
        current_input = x
        for layer in self.layers:
            current_input = layer.forward(current_input)
        return current_input

    def summary(self):
        """
        Prints a diagnostic table summarizing the architecture and capacity of the network.
        """
        print("=" * 60)
        print(f"{'Layer (Type)':<20} {'Shape (In, Out)':<20} {'Param #':<15}")
        print("=" * 60)
        
        total_params = 0
        for i, layer in enumerate(self.layers):
            layer_type = f"Dense-{i+1} ({layer.activation_name})"
            shape_str = f"({layer.weights.shape[0]}, {layer.weights.shape[1]})"
            
            layer_params = layer.weights.size + layer.biases.size
            total_params += layer_params
            
            print(f"{layer_type:<20} {shape_str:<20} {layer_params:<15}")
            print("-" * 60)
            
        print(f"Total params: {total_params}")
        print("=" * 60)

    def save_model(self, filepath: str = "../models/mlp_model.json"):
        """
        Serializes the network topology and learned parameters to disk.
        """
        model_data = {
            "topology": [self.layers[0].weights.shape[0]] + [layer.weights.shape[1] for layer in self.layers],
            "layers": []
        }
        
        for layer in self.layers:
            layer_data = {
                "activation": layer.activation_name,
                "weights": layer.weights.tolist(),
                "biases": layer.biases.tolist()
            }
            model_data["layers"].append(layer_data)
            
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=4)
        print(f"Model successfully saved to {filepath}")
        
    def backward(self, loss_grad: np.ndarray, learning_rate: float):
        """
        Executes the backpropagation pass to update weights and biases based on the error.
        
        Why backwards?
        The Chain Rule requires us to start with the error at the final output (loss_grad) 
        and pass the derivative signal backwards through each layer. Each layer uses the 
        gradient from the layer ahead of it to calculate its own weight/bias adjustments, 
        and then computes the gradient for the layer behind it.
        """
        current_gradient = loss_grad
        
        # Iterate through layers in reverse order 
        for layer in reversed(self.layers):
            current_gradient = layer.backward(current_gradient, learning_rate)
