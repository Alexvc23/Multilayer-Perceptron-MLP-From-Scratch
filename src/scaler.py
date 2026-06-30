import numpy as np
import json
import os

def encode_labels(y)-> np.ndarray:
    """
    Converts categorical string labels into numerical binary targets.
    
    Why: Multi-layer perceptrons require mathematical tensors for their matrix 
    multiplications. Also, calculating errors with Binary Cross-Entropy requires 
    ground truth targets to be strictly numerical values of 1 or 0 (probability).
    
    Args:
        y (np.ndarray or list): Array of nominal labels (e.g., 'M' and 'B').
        
    Returns:
        np.ndarray: Array of shape (n_samples,) containing 1s and 0s.
    """
    # Using np.where to efficiently map 'M' to 1 and 'B' to 0 across the entire array
    return np.where(np.array(y) == 'M', 1, 0)


class StandardScaler:
    """
    Stateful Scaler that standardizes features by removing the mean and scaling to unit variance.
    Calculates parameters (mean and std) exclusively on training data to prevent data leakage.
    """
    def __init__(self):
        # State variables to hold the mean and standard deviation matrices
        self.mean_ = None
        self.std_ = None

        # Epsilon prevents division-by-zero if a feature has zero variance (constant value)
        self.epsilon = 1e-8

    def fit(self, X):
        """
        Calculates and stores the mean and standard deviation for each feature.
        
        Why: We calculate this only once on the training data so that when we 
        evaluate performance on the validation set, the validation set is 
        transformed using the exact same metrics the model learned from.
        This explicitly prevents "data leakage".
        
        Args:
            X (np.ndarray): Training inputs of shape (n_samples, n_features)
        """
        X = np.array(X, dtype=float)

        # Calculate mean across rows (axis=0) to get average for each feature (column)
        # Math: mu = 1/N * sum(x_i)
        self.mean_ = np.mean(X, axis=0)

        # Calculate standard deviation across rows (axis=0)
        # Math: sigma = sqrt( 1/N * sum((x_i - mu)^2) )
        self.std_ = np.std(X, axis=0)

    def transform(self, X):
        """
        Applies standard scaling to the dataset using the previously fitted mean and std.
        
        Why: Standardizes the input magnitudes so gradient descent converges smoothly.
        Without this, features with larger magnitudes dominate the gradient updates.
        
        Args:
            X (np.ndarray): Data to transform, shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Scaled data, shape (n_samples, n_features)
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted yet. Call fit() first.")

        X = np.array(X, dtype=float)

        # Math: z = (x - mu) / (sigma + epsilon)
        # Vectorized operation applies the formula to the whole matrix efficiently
        X_scaled = (X - self.mean_) / (self.std_ + self.epsilon)

        return X_scaled

    def save(self, filepath):
        """
        Serializes the learned mean and standard deviation to a JSON file.
        
        Why: Allows exactly preserving the transformation state so the separate 
        prediction script can scale new patient data equivalently.
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler is not fitted, nothing to save.")

        data = {
            # Convert NumPy arrays to lists since JSON cannot natively serialize ndarrays
            'mean': self.mean_.tolist(),
            'std': self.std_.tolist()
        }

        os.makedirs(os.path.dirname(os.path.abspath(filepath)) or '.', exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)

    # ──────────────────────────────────────────────────────────────────────────────

    def load(self, filepath = "models/scaler.json"):
        """
        Deserializes the mean and standard deviation from a JSON file.
        
        Why: Allows instantiating an already-fitted scaler for inference/prediction.

        Loads the starndardization parameters (mean and std dev) from a json file.

        Why we load these paremeters instead of recalculating them :
        # ──────────────────────────────────────────────────────────
        To prevent data laeakage. In Phase 4 (Prediction/Validation), we must
        transform the new data using the exact same Mean (mu) and Standard Deviation
        (sigma) that were calculated from the training data in Phase 3. This ensures
        that the model's predictions are based on the same feature scaling it was trained on

        Parameters:
            filepath (str): Path to the JSON file containing the scaler parameters.
        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the mean and standard deviation arrays.
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Parse standard Python lists back into mathematical vectors (numpy arrays)
            self.mean_ = np.array(data['mean'], dtype=float)
            self.std_ = np.array(data['std'], dtype=float)

        except FileNotFoundError as e:
            print(f"File not found. Ensure training was completed: {e}")
            return np.array([]), np.array([])

        except KeyError as e:
            print(f"Missing key in scaler JSON : {e}")
            return np.array([]), np.array([])

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON. Ensure the file is valid: {e}")
            return np.array([]), np.array([])

# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test Block to prove mathematical correctness (can be run directly with `python src/scaler.py`)
    print("--- Testing StandardScaler & Encode Labels ---")
    
    # Test Label Encoding
    dummy_labels = np.array(['B', 'M', 'B', 'B', 'M'])
    encoded = encode_labels(dummy_labels)
    assert np.all(encoded == np.array([0, 1, 0, 0, 1]))
    print("[OK] Label Encoding (M=1, B=0)")
    
    # Test Standardization
    dummy_X = np.array([
        [10.0, 100.0, 0.5],
        [12.0,  90.0, 0.6],
        [ 8.0, 110.0, 0.4],
        [10.0, 100.0, 0.5]
    ])
    
    scaler = StandardScaler()
    scaler.fit(dummy_X)
    X_scaled = scaler.transform(dummy_X)
    
    # Math proofs
    assert np.allclose(np.mean(X_scaled, axis=0), 0)
    assert np.allclose(np.std(X_scaled, axis=0), 1)
    print("[OK] Standardization (mean ~0.0, std ~1.0)")
    
    # Test Persistence
    test_file = "test_scaler_persistence.json"
    scaler.save(test_file)
    
    scaler_loaded = StandardScaler()
    scaler_loaded.load(test_file)
    assert np.allclose(scaler.mean_, scaler_loaded.mean_)
    assert np.allclose(scaler.std_, scaler_loaded.std_)
    print("[OK] State Persistence (save/load via JSON)")
    
    if os.path.exists(test_file):
        os.remove(test_file)
    
    print("--- All scaler unit tests passed! ---")
