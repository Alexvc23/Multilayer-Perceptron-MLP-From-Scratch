"""
Loss Functions

Implements the Binary Cross-Entropy loss formula required to evaluate
classification errors on the output layer.
"""
import numpy as np

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the exact BCE formula: E = - (1/N) * sum( y*log(y_pred) + (1-y)*log(1-y_pred) )
    
    Why the negative sign? 
    Probabilities range from 0 to 1, causing the log to be negative. We want loss 
    to be a positive cost, so we multiply by -1.
    
    Numerical Stability:
    We use np.clip to constrain predictions between 1e-15 and 1 - 1e-15. This prevents
    taking log(0), which mathematically evaluates to undefined (or -inf) and would crash the network.
    
    This cost function heavily penalizes the network when it is "confidently wrong"
    (e.g., predicting 0.99 for a Benign case).
    """
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    N = y_true.shape[0]
    
    # Exact BCE formula applying penalty based on truth value
    loss = - (1 / N) * np.sum(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
    return float(loss)

def binary_cross_entropy_prime(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculates the initial derivative (gradient) of BCE, to jumpstart backpropagation.
    
    This function calculates the local gradient for the output layer to determine 
    how much each prediction contributed to the output error during backpropagation.
    Math: dE/dp = (1/N) * ((1-y)/(1-p) - (y/p))
    """
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    N = y_true.shape[0]
    
    # Compute gradient: how much did this specific probability prediction contribute to the error?
    term1 = (1 - y_true) / (1 - y_pred_clipped)
    term2 = y_true / y_pred_clipped
    
    gradient = (term1 - term2) / N
    return gradient

if __name__ == "__main__":
    # Manual testing to prove the "Confidently Wrong" penalty scaling for defense readiness
    y = np.array([1, 0])
    p_good = np.array([0.99, 0.01])
    p_bad = np.array([0.01, 0.99])
    
    loss_good = binary_cross_entropy(y, p_good)
    loss_bad = binary_cross_entropy(y, p_bad)
    
    print("--- Forward Loss Calculation ---")
    print(f"Target Vector (Malignant, Benign): {y}")
    print(f"Loss for GOOD predictions {p_good}: {loss_good:.4f}")
    print(f"Loss for BAD predictions  {p_bad}: {loss_bad:.4f}")
    
    print("\n--- Backward Pass (Gradients) ---")
    grad_good = binary_cross_entropy_prime(y, p_good)
    grad_bad = binary_cross_entropy_prime(y, p_bad)
    print(f"Gradient for GOOD predictions:\n{grad_good}")
    print(f"Gradient for BAD predictions:\n{grad_bad}")
    print("\nInsight: Notice the massive gradients (-50 and 50) for bad predictions.")
    print("This forces drastic weight updates during gradient descent compared to good predictions.")

