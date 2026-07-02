# GitHub Copilot Instructions: MLP from Scratch (Wisconsin Breast Cancer)

## 1. Project Objective & Context
You are assisting in an **educational machine learning project** where a Multilayer Perceptron (MLP) is being built **entirely from scratch** in Python. The goal is to classify the Wisconsin breast cancer dataset (Malignant "M" vs. Benign "B"). Success is defined by the developer’s ability to verbally defend the underlying mathematics to a human evaluator.

## 2. The "No-Magic" Rule (Strict Prohibition)
**STRICTLY FORBIDDEN:** You must never suggest, import, or use "black-box" machine learning or deep learning libraries. This includes, but is not limited to:
*   TensorFlow / Keras
*   PyTorch
*   Scikit-learn (specifically `MLPClassifier`, `train_test_split`, or preprocessing modules)
*   XGBoost / LightGBM

**Every algorithm—including feedforward, backpropagation, and gradient descent—must be implemented manually.**

## 3. Permitted Toolset
You may **ONLY** suggest code utilizing the following libraries:
*   **Core Logic:** Standard Python libraries.
*   **Linear Algebra:** `numpy` (specifically for vectorized operations and matrix multiplication).
*   **Data Handling:** `pandas` (strictly for CSV ingestion and basic manipulation).
*   **Visualization:** `matplotlib` or `seaborn` (strictly for plotting the mandatory learning curves: Loss and Accuracy).

## 4. Coding Standards for Defense Preparedness
Every suggestion must be optimized for a **human-evaluated defense**.
*   **Docstring Requirement:** Every function **must** include a docstring explaining the "Why" behind the logic. Focus on its role in the learning phase (e.g., "This function calculates the local gradient for the hidden layer to determine how much each weight contributed to the output error during backpropagation").
*   **Mathematical Transparency:** Do not hide math inside complex abstractions. When implementing core formulas, write them out clearly using `numpy`. 
    *   **Weighted Sum:** $z = \sum (x_k \cdot w_k) + bias$.
    *   **Binary Cross-Entropy:** Implement the exact formula: $E = -\frac{1}{N} \sum [y_n \log(p_n) + (1 - y_n) \log(1 - p_n)]$.
    *   **Softmax:** Implement manually for the output layer to provide a probabilistic distribution.
*   **Inline Explanations:** Include inline comments for derivatives (e.g., Sigmoid prime) so the developer can explain the calculus during evaluation.

## 5. Architectural Requirements & Modularity
*   **Atomic Functions:** Keep functions small and modular. Ensure separate functions for:
    *   Weight/Bias initialization (using `np.random.seed` for repeatability).
    *   Forward pass (Linear step -> Activation step).
    *   Loss calculation (Binary Cross-Entropy).
    *   Backward pass (Backpropagation of gradients).
    *   Weight updates (Gradient Descent).
*   **Layer Constraints:** Ensure the default architecture supports at least **two hidden layers**.
*   **Data Integrity:** Always include logic to split the raw data into separate **training and validation sets** to test against unknown examples.

## 6. Project Phases Reference
Align your suggestions with the following operational phases:
1.  **Data Splitter:** Program to partition the 32-column dataset.
2.  **Training Program:** Program that implements the full learning loop and saves the model (topology and weights) to a file.
3.  **Prediction Program:** Program that loads the saved model and evaluates performance on the validation set.