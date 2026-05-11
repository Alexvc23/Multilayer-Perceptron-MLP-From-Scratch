# Multilayer Perceptron from Scratch

This project implements a Multilayer Perceptron (MLP) entirely from scratch using only Python, NumPy, Pandas, and Matplotlib. It is designed to classify the Wisconsin breast cancer dataset (Malignant vs. Benign).

## Project Structure

```text
.
├── Makefile              # Commands to execute the different phases (split, train, predict)
├── README.md             # Project documentation and structure
├── data/                 # Directory containing raw and partitioned dataset files
├── models/               # Directory where the trained model layout and weights are saved
├── split.py              # Entry script to partition data
├── train.py              # Entry script for the learning phase (training loop)
├── predict.py            # Entry script to load model and evaluate on validation data
└── src/                  # Core mathematical engine
    ├── activations.py    # Non-linear activation functions (Sigmoid, Softmax) and their derivatives
    ├── layer.py          # Dense layer implementation (weights, biases, local gradients)
    ├── loss.py           # Binary Cross-Entropy loss calculation
    ├── metrics.py        # Evaluation metrics like accuracy
    └── network.py        # MultilayerPerceptron orchestrator class
```

## Mathematical Engine & Core Responsibilities

The core of the MLP is housed entirely within the `src/` directory to cleanly separate execution parsing from pure mathematics.

### Feedforward Pass
The forward pass is initiated in `src/network.py` and delegates computation down to each layer:
1. **`src/layer.py`**: Computes the weighted sum $z = \sum (x_k \cdot w_k) + bias$.
2. **`src/activations.py`**: Applies non-linearity to the linear output $z$. For hidden layers, this is typically the Sigmoid function; for the output layer, it will be Softmax to provide a probabilistic distribution.
3. **`src/loss.py`**: At the end of the forward pass, the predictions are compared to the actual targets using Binary Cross-Entropy to quantify the network's error.

### Backpropagation Pass
The backward pass distributes the error backwards through the network to update the parameters:
1. **`src/loss.py`**: Determines the initial gradient of the loss with respect to the network's final output.
2. **`src/layer.py`**: Computes the local gradients. It calculates how much each weight and bias contributed to the error.
3. **`src/activations.py`**: Supplies the derivatives of the activation functions (e.g., Sigmoid prime) needed by the chain rule to pass the gradient through the non-linearities.
4. **`src/network.py`**: Orchestrates the chain rule across all layers and applies Gradient Descent to update the weights and biases based on the calculated gradients.

## Constraints & "No-Magic" Rule
This codebase strictly prohibits black-box machine learning libraries (TensorFlow, PyTorch, Scikit-Learn, etc.).
- **NumPy** is used for dot products and vectorized math.
- **Pandas** is used purely for CSV ingestion and base manipulation.
- **Matplotlib** is used exclusively for plotting the learning curves (loss and accuracy over epochs).