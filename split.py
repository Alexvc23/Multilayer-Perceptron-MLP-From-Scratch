"""
Data Splitter

This script is responsible for partitioning the raw Wisconsin breast cancer dataset
into a training set and a validation set. It strictly uses pandas for ingestion
and manipulation.

Instructional Objectives:
1. No high-level ML libraries (e.g., scikit-learn).
2. Uses pseudo-random seed mechanism for identical splits.
3. Keep features and labels perfectly synchronized.
4. Save resulting partitions into training_data.csv and validation_data.csv.
"""
import pandas as pd
import numpy as np
import argparse
import os

def split_data(df: pd.DataFrame, test_ratio: float = 0.2, seed: int = 42):
    """
    Splits a pandas DataFrame into training and validation sets without using 
    external ML libraries like scikit-learn.

    Args:
        df (pd.DataFrame): The raw dataset.
        test_ratio (float): The proportion of the dataset to include in the validation split.
        seed (int): The random seed to ensure reproducibility.

    Returns:
        tuple: (train_df, validation_df)
    
    Why we do this:
        - Validation Set: We need a separate validation set to detect overfitting. 
          By testing the model on unseen data, we ensure it learns generalized features 
          rather than just memorizing the training data.
        - Random Seed: Using a fixed random seed is critical for the scientific method.
          It guarantees reproducibility, meaning that any evaluator or researcher running 
          this code will get the exact same data split.
        - Manual Shuffling: Shuffling the raw dataframe before splitting ensures that 
          the features ($X$) and the labels ($y$) remain perfectly synchronized.
    """
    # Set seed for reproducibility. It initializes the random number generator.
    np.random.seed(seed)
    
    # Generate a random permutation of row indices.
    # This shuffles the indices without changing the actual data yet.
    shuffled_indices = np.random.permutation(len(df))
    
    # Calculate the exact index where the split should occur based on the test_ratio.
    test_size = int(len(df) * test_ratio)
    
    # Slice the shuffled indices to separate the validation indices and training indices.
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]
    
    # Select the rows using the sliced indices to form the final DataFrames.
    # We reset the index to provide a clean continuous index for down-stream tasks.
    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[test_indices].reset_index(drop=True)
    
    return train_df, val_df

def main():
    parser = argparse.ArgumentParser(description="Split dataset into training and validation sets.")
    parser.add_argument("data_path", type=str, nargs="?", default="data/data.csv", help="Path to raw dataset.")
    parser.add_argument("--ratio", type=float, default=0.2, help="Validation set ratio (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.data_path}...")
    try:
         # Read the CSV without headers as seen during data exploration.
        df = pd.read_csv(args.data_path, header=None)
    except FileNotFoundError:
        print(f"Error: Could not find dataset at {args.data_path}")
        return

    print("Splitting data...")
    train_df, val_df = split_data(df, test_ratio=args.ratio, seed=args.seed)
    
    # Construct output paths relative to the input file directory, or defaults
    output_dir = os.path.dirname(args.data_path)
    train_path = os.path.join(output_dir, "training_data.csv") if output_dir else "training_data.csv"
    val_path = os.path.join(output_dir, "validation_data.csv") if output_dir else "validation_data.csv"
    
    print(f"Saving training data ({len(train_df)} rows) to {train_path}...")
    train_df.to_csv(train_path, index=False, header=False)
    
    print(f"Saving validation data ({len(val_df)} rows) to {val_path}...")
    val_df.to_csv(val_path, index=False, header=False)
    
    print("Data splitting complete!")

if __name__ == "__main__":
    main()
