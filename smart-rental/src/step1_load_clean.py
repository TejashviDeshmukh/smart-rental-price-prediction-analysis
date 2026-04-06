# Step 1: Load and Clean the Rental Dataset
# ------------------------------------------
# Goal: Import the data, understand its shape, and fix any issues
# like missing values or impossible numbers.

import pandas as pd
import numpy as np


def load_and_clean(filepath="data/rental_data.csv"):
    # Load the dataset into a DataFrame (a table-like structure in Python)
    df = pd.read_csv(filepath)

    print("=== Raw Data Overview ===")
    print(df.head())              # first 5 rows
    print("\nShape:", df.shape)   # (rows, columns)
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum())

    # Drop rows where Rent or Size is missing — these are essential columns
    df.dropna(subset=["Rent", "Size"], inplace=True)

    # Fill missing BHK with the most common value (mode)
    df["BHK"].fillna(df["BHK"].mode()[0], inplace=True)

    # Fill missing Furnishing Status with 'Unfurnished'
    df["Furnishing Status"].fillna("Unfurnished", inplace=True)

    # Remove rows with impossible values (rent or size can't be 0 or negative)
    df = df[df["Rent"] > 0]
    df = df[df["Size"] > 0]
    df = df[df["BHK"] > 0]

    # Reset index after dropping rows so numbering is clean (0, 1, 2, ...)
    df.reset_index(drop=True, inplace=True)

    print("\n=== After Cleaning ===")
    print("Shape:", df.shape)
    print(df.describe())  # summary statistics

    return df


if __name__ == "__main__":
    df = load_and_clean()
    print("\nData loaded and cleaned successfully!")
