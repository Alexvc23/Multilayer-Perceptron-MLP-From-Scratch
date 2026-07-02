"""
Training Program

This script:
1) initializes the Multilayer Perceptron
2) runs the training loop using feedforward and backpropagation
3) plots the learning curves (loss and accuracy)
and persists the trained topology and weights to the models/ directory.
"""
from ast import arg
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from src.network import MultilayerPerceptron
from src.scaler import StandardScaler, encode_labels
from src.loss import categorical_cross_entropy


def train_model(hidden_layers: list = [24, 24], epochs: int = 6000, learning_rate: float = 0.7):
    """
    Executes the training phase on the partitioned training data.

    Why: This is the core learning loop. It coordinates data loading, scaling,
    forward passes, error calculation (BCE), and the crucial backpropagation
    step where weights are updated via Gradient Descent (Task 5).
    """
    # 1. Load Data
    try:
        train_df = pd.read_csv("data/training_data.csv", header=None)
        val_df = pd.read_csv("data/validation_data.csv", header=None)
    except FileNotFoundError:
        print("Error: Partitioned data not found. Please run split.py first.")
        return

    # Slice data: Col 0 is ID, Col 1 is Label, Col 2-31 are features
    X_train_raw = train_df.iloc[:, 2:].values
    y_train_raw = train_df.iloc[:, 1].values
    X_val_raw = val_df.iloc[:, 2:].values
    y_val_raw = val_df.iloc[:, 1].values

    # 2. Preprocessing
    # Standardize features (Mean=0, Std=1) to ensure smooth gradient descent convergence.
    scaler = StandardScaler()
    scaler.fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    scaler.save("models/scaler.json")

    # Encode labels as one-hot vectors so they match the two-unit softmax output.
    y_train = encode_labels(y_train_raw)
    y_val = encode_labels(y_val_raw)

    # 3. Initialize Network
    # Topology: [Input (30 features), Hidden1 (24), Hidden2 (24), Output (2)]
    # We use two hidden layers to satisfy architectural requirements.
    topology = [X_train.shape[1]] + hidden_layers + [2]
    mlp = MultilayerPerceptron(
        topology, hidden_activation="relu", output_activation="softmax"
    )
    mlp.summary()

    # 4. Hyperparameters
    epochs = epochs
    learning_rate = learning_rate
    history = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}

    print(f"\nStarting training for {epochs} epochs...")

    # 5. Training Loop
    for epoch in range(epochs):
        # --- Forward Pass ---
        y_pred_train = mlp.forward(X_train)
        train_loss = categorical_cross_entropy(y_train, y_pred_train)

        # --- Backward Pass (Weight Updates) ---
        # For softmax + categorical cross-entropy, the output-layer gradient collapses to dZ = P - Y.
        # This is the exact signal the output layer needs before the chain rule continues backward.
        loss_grad = y_pred_train - y_train
        # mlp.backward propagates this simplified gradient and updates weights via Gradient Descent.
        mlp.backward(loss_grad, learning_rate)

        # --- Validation & Metrics ---
        y_pred_val = mlp.forward(X_val)
        val_loss = categorical_cross_entropy(y_val, y_pred_val)

        # Calculate accuracy for tracking
        train_acc = np.mean((y_pred_train > 0.5) == y_train)
        val_acc = np.mean((y_pred_val > 0.5) == y_val)

        # Log history
        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch:4d}/{epochs} | Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

    # 6. Finalization
    # Save the trained model's architecture and weights for future inference or analysis.
    os.makedirs("models", exist_ok=True)
    mlp.save_model("models/mlp_model.json")

    # Visualize training progress with learning curves to diagnose overfitting and confirm training dynamics.
    plot_learning_curves(history)


def plot_learning_curves(history: dict):
    """
    Visualizes the training progress.

    Why: Mandatory learning curves help diagnose overfitting (when val_loss starts
    increasing while train_loss decreases) and confirm gradient descent is working.
    """
    epochs = range(len(history["loss"]))

    plt.figure(figsize=(12, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.title("Binary Cross-Entropy Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.title("Classification Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("learning_curves.png")
    print("\nLearning curves saved to learning_curves.png")
    plt.show()


if __name__ == "__main__":
    print("Running training phase...")

    parser = argparse.ArgumentParser(description="Train Neural Network")

    # 1. set up hidden layer sizes as a list of integers 
    # Handle the list (nargs='+' allows multiple numbers)
    parser.add_argument('--hidden', nargs='+', type=int, default=[24, 24], 
                        help='List of hidden layer sizes (e.g. --hidden 24 24)')
    
    # 2. set up epochs as an integer 
    # Handle the integers
    parser.add_argument('--epochs', type=int, default=6000)

    # 3. set up learning rate as a float
    # Handle the floats
    parser.add_argument('--lr', type=float, default=0.7, dest='learning_rate')

    args = parser.parse_args()

    # Call the training function with parsed arguments
    train_model(
        hidden_layers=args.hidden, 
        epochs=args.epochs, 
        learning_rate=args.learning_rate
    )
