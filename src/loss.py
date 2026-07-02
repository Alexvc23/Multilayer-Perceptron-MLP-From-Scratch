"""
Loss Functions

Implements the categorical cross-entropy loss formula required to evaluate
classification errors on the output layer.
"""
import numpy as np


def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the categorical cross-entropy loss for one-hot encoded targets.

    Why: A 2-neuron softmax output produces a probability distribution across
    classes. Categorical cross-entropy measures how far that distribution is
    from the one-hot encoded truth labels.

    Formula:
        E = - (1/N) * sum_i sum_c y[i, c] * log(p[i, c] + epsilon)

    Numerical Stability:
        We clip the predictions to avoid log(0), which would otherwise produce
        negative infinity and break training.
    """
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    N = y_true.shape[0]
    loss = - (1 / N) * np.sum(y_true * np.log(y_pred_clipped))
    return float(loss)


def categorical_cross_entropy_prime(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculates the raw categorical cross-entropy gradient with respect to predictions.

    Why: This returns the direct loss gradient before the softmax simplification
    is applied in the backward pass.
    """
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    #shape[0] means the number of samples in the batch, which is used to normalize the gradient.
    N = y_true.shape[0]
    return -(y_true / y_pred_clipped) / N


if __name__ == "__main__":
    # Manual testing to prove the categorical loss and gradient behavior.
    y = np.array([[1, 0], [0, 1]])
    p_good = np.array([[0.99, 0.01], [0.01, 0.99]])
    p_bad = np.array([[0.01, 0.99], [0.99, 0.01]])

    loss_good = categorical_cross_entropy(y, p_good)
    loss_bad = categorical_cross_entropy(y, p_bad)

    print("--- Forward Loss Calculation ---")
    print(f"Target Matrix (one-hot):\n{y}")
    print(f"Loss for GOOD predictions:\n{p_good}\n{loss_good:.4f}")
    print(f"Loss for BAD predictions:\n{p_bad}\n{loss_bad:.4f}")

    print("\n--- Backward Pass (Gradients) ---")
    grad_good = categorical_cross_entropy_prime(y, p_good)
    grad_bad = categorical_cross_entropy_prime(y, p_bad)
    print(f"Gradient for GOOD predictions:\n{grad_good}")
    print(f"Gradient for BAD predictions:\n{grad_bad}")
    print("\nInsight: The gradient is now expressed per class in the one-hot output space.")
    print("The final softmax + cross-entropy simplification will be applied in the backward pass.")

