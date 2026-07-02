"""
Prediction Program

This script loads the strictly trained network topology and weights from the models/ folder,
runs a forward pass over the validation dataset, and evaluates performance (loss and confusion matrix).
"""

import json
import pandas as pd
import numpy as np
import colorama
from src.network import MultilayerPerceptron
from src.scaler import StandardScaler, encode_labels
from src.loss import categorical_cross_entropy


def load_model(filepath: str = "models/mlp_model.json") -> dict:
    """
    Safetily load a neural network's topology and waights from a JSON file.
    Parameters:
        filepath (str): Path to the JSON file containing the model's configuration
    Returns:
        dict: The parsed model data or empty dict if loading fails
    """
    try:
        # utf-8 encoding ensures we can read files with special characters without issues.
        with open(filepath, "r", encoding="utf-8") as f:
            model_data = json.load(f)
        return model_data

    except FileNotFoundError as e:

        print(f"File not found : {e}")
        return {}

    except json.JSONDecodeError as e:

        print(f"Error decoding JSON : {e}")
        return {}

    except Exception as e:

        print(f"An unexpected error occurred : {e}")
        return {}


# ──────────────────────────────────────────────────────────────────────────────

def predict_model():
    """
    Loads model weights and executes forward pass on validation set.
    """
    # av - we load the trained model from disk
    loaded_model = load_model()
    print(
        f"Loaded model topology: {loaded_model.get('topology', 'No topology found')}\n\n"
    )

    # av - we load the scaler parameters from disk
    # in order to standardize the validation data in the same
    # way as the training data was standardized

    # av - initialize the scaler class to store the mean and std values
    scaler = StandardScaler()
    # av - we load the train scaler parameters from disk to ensure
    # the validation data is standardized in the same way
    scaler.load(filepath="models/scaler.json")

    # # av - we load the validation dataset
    try:
        val_raw_data = pd.read_csv("./data/validation_data.csv", header=None)
        print(f"row validation data: \n{val_raw_data}\n")
    except FileNotFoundError as e:
        print(f"Validation data file not found: {e}")
        return
    # av - slice the data in to features(x) and target_labels(y)
    # col 0 is the id, col 1 is the target label, and the rest 2-31 the features
    # take all the rows, and columns for index 2 to the end (features)
    x_val  = val_raw_data.iloc[:,2:].values
    # take all the rows from the target variable
    y_val = val_raw_data.iloc[:, 1].values

    # av - apply standardization to the validation features using the loaded scaler parameters
    x_val = scaler.transform(x_val)

    # One-hot encode labels so they match the softmax output shape.
    y_val_encoded = encode_labels(y_val)

    # av - initialize network classe to store our previously
    # loaded model with its weights and biases ect
    model = MultilayerPerceptron(
        topology=loaded_model["topology"],
        hidden_activation="relu",
        output_activation="softmax"
    )
    print(
        colorama.Style.BRIGHT
        + colorama.Fore.GREEN
        + f"\n MultilayePerceptron initialized: \n"
        + colorama.Style.RESET_ALL
    )
    model.summary()

    # av - feed the initialized classe with the trained model parameters
    for layer, layer_data in zip(model.layers, loaded_model["layers"]):
        layer.weights = np.array(layer_data["weights"])
        layer.biases = np.array(layer_data["biases"])

    # ===============================================================================
    #             Preforming the predictions part
    # ===============================================================================

    np.set_printoptions(precision=4, suppress=True)

    # av - we carry out the forward passs on the validation data (unknown data) to get the predictions
    y_pred_val = model.forward(x_val)

    print(f"Shape of predicted values: {y_pred_val.shape}")

    # av - we calculate the loss on the validation data using categorical cross entropy
    val_loss = categorical_cross_entropy(y_val_encoded, y_pred_val)
    val_pred_classes = np.argmax(y_pred_val, axis=1)
    val_true_classes = np.argmax(y_val_encoded, axis=1)
    val_acc = np.mean(val_pred_classes == val_true_classes)

    print(
        colorama.Style.BRIGHT + colorama.Fore.GREEN
        + " \nExample with the last 5 predictions\n"
        + colorama.Style.RESET_ALL
    )
    print(f"Real data labels: \n{y_val[:5]}\n{y_val_encoded[:5].tolist()}\n")
    print(f"Predicted probabilities: \n{(y_pred_val[:5].tolist())}\n")
    print(
        f"Predicted classes via argmax: \n{np.argmax(y_pred_val[:5], axis=1).tolist()}\n"
    )
    print(
        f"{colorama.Style.BRIGHT}Validation Accuracy for the entire dataset:{colorama.Style.BRIGHT + colorama.Fore.GREEN} {val_acc:.4f}"
    )

if __name__ == "__main__":
    print("Running prediction phase...\n\n")
    predict_model()
