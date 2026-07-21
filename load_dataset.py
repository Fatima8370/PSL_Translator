"""
Walks the entire dataset/ directory tree and combines all CSVs into two
NumPy arrays: X (features) and y (string labels).

Design A implementation: the dialect_X folder name is used ONLY to locate
files during collection. It is explicitly discarded here — a "B" from
dialect_1 and a "B" from dialect_2 both become the same class "B".
"""

import os
import numpy as np
import pandas as pd

import config


def load_full_dataset():
    X_list = []
    y_list = []

    dataset_root = config.DATASET_ROOT

    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"No dataset folder found at '{dataset_root}'.")
    else:
        pass

    dialect_folders = os.listdir(dataset_root)

    for dialect_folder in dialect_folders:
        dialect_path = os.path.join(dataset_root, dialect_folder)

        if not os.path.isdir(dialect_path):
            # Skip stray non-folder files sitting in dataset/ (e.g. .DS_Store)
            continue
        else:
            pass

        csv_files = [f for f in os.listdir(dialect_path) if f.endswith(".csv")]

        for csv_file in csv_files:
            # The label is the filename without extension, e.g. "A.csv" -> "A".
            # Dialect folder name is read only to locate the file, then discarded.
            label = os.path.splitext(csv_file)[0]
            csv_path = os.path.join(dialect_path, csv_file)

            df = pd.read_csv(csv_path)

            if df.empty:
                print(f"[WARNING] {csv_path} is empty. Skipping.")
                continue
            else:
                pass

            X_list.append(df.values)
            y_list.append(np.full(shape=(df.shape[0],), fill_value=label))

    if len(X_list) == 0:
        raise ValueError("No data found across any dialect folder. Collect some samples first.")
    else:
        pass

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    return X, y


if __name__ == "__main__":
    # Quick standalone sanity check — run this file directly to inspect
    # what's actually been collected so far before training.
    X, y = load_full_dataset()
    unique_labels, counts = np.unique(y, return_counts=True)

    print(f"Total samples: {X.shape[0]}")
    print(f"Feature vector length: {X.shape[1]}")
    print("Samples per label (combined across dialects):")
    for label, count in zip(unique_labels, counts):
        print(f"  {label}: {count}")