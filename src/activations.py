"""
Activation Functions

Contains mathematical implementations for non-linear activations and their derivatives,
essential for computing the chain rule during backpropagation.
"""
import numpy as np

def relu(z: np.ndarray) -> np.ndarray:
    """
    Forward pass for ReLU (Rectified Linear Unit).
    
    Why: Used primarily in hidden layers to prevent the vanishing gradient problem.
    It allows the network to learn non-linear patterns while being computationally efficient.
    
    Mathematics: f(z) = max(0, z)
    """
    return np.maximum(0, z)

def relu_prime(z: np.ndarray) -> np.ndarray:
    """
    Backward pass (derivative) for ReLU.
    
    Why: Determines how much each weight contributed to the output error during backpropagation.
    Unlike sigmoid, its derivative is exactly 1 for positive inputs, ensuring gradients
    flow back cleanly without diminishing recursively through the layers.
    
    Mathematics: f'(z) = 1 if z > 0 else 0
    """
    return np.where(z > 0, 1.0, 0.0)

def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Forward pass for Sigmoid.
    
    Why: Squashes output to a range of (0, 1). This is ideal for binary classification
    output layers, as the output can be interpreted as a probability (e.g., Malignant vs Benign).
    
    Mathematics: f(z) = 1 / (1 + e^-z)
    """
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z: np.ndarray) -> np.ndarray:
    """
    Backward pass (derivative) for Sigmoid.
    
    Why: Used in backpropagation to calculate the local gradient. Note that this derivative
    approaches 0 for very large or very small inputs, which can cause the vanishing gradient
    problem if stacked in deep hidden layers.
    
    Mathematics: f'(z) = f(z) * (1 - f(z))
    """
    s = sigmoid(z)
    return s * (1.0 - s)

def softmax(z: np.ndarray) -> np.ndarray:
    """
    Forward pass for Stable Softmax.
    
    Why: Turns a vector of raw logits into a valid multiclass probability distribution
    that sums to 1. 
    
    Numeric Stability: Subtracting the maximum of z before applying np.exp() prevents 
    overflow crashes with large positive logits.
    
    Mathematics: f(z)_i = e^(z_i - max(z)) / sum(e^(z_j - max(z)))
    """
    shifted_z = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(shifted_z)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

if __name__ == "__main__":
    test_vec = np.array([-2.0, 0.0, 2.0])
    
    print("--- Input ---")
    print(f"z: {test_vec}\n")

    print("--- ReLU ---")
    print(f"ReLU(z):       {relu(test_vec)}")
    print(f"ReLU_prime(z): {relu_prime(test_vec)}\n")

    print("--- Sigmoid ---")
    print(f"Sigmoid(z):       {sigmoid(test_vec)}")
    print(f"Sigmoid_prime(z): {sigmoid_prime(test_vec)}\n")

    print("--- Softmax ---")
    print(f"Softmax(z):       {softmax(test_vec)}")
    print(f"Sum of Softmax:   {np.sum(softmax(test_vec))}")
