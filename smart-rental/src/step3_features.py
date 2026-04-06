# Step 3: Feature Engineering & Encoding
# ----------------------------------------
# Goal: Create useful new features and convert text columns to numbers.
# ML models only understand numbers — not words like "Mumbai" or "Furnished".

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from step1_load_clean import load_and_clean


def engineer_features(df):
    # --- New Feature: Price per Square Foot ---
    # Normalizes rent by size so we can compare properties fairly.
    # A ₹30,000 rent for 600 sqft is much more expensive than for 1500 sqft.
    df = df.copy()
    df["Price_per_sqft"] = df["Rent"] / df["Size"]

    # --- Label Encoding: City ---
    # Converts each unique city name to a unique integer.
    # e.g., Bangalore=0, Chennai=1, Delhi=2, Hyderabad=3, Mumbai=4
    le_city = LabelEncoder()
    df["City_encoded"] = le_city.fit_transform(df["City"])

    # --- Label Encoding: Furnishing Status ---
    # e.g., Furnished=0, Semi-Furnished=1, Unfurnished=2
    le_furnish = LabelEncoder()
    df["Furnishing_encoded"] = le_furnish.fit_transform(df["Furnishing Status"])

    print("=== Label Encoding Mappings ===")
    print("Cities   :", dict(zip(le_city.classes_,
                                 le_city.transform(le_city.classes_).tolist())))
    print("Furnishing:", dict(zip(le_furnish.classes_,
                                  le_furnish.transform(le_furnish.classes_).tolist())))

    # --- Select Features (inputs) and Target (output) ---
    features = ["BHK", "Size", "City_encoded", "Furnishing_encoded", "Price_per_sqft"]
    X = df[features]   # inputs the model learns from
    y = df["Rent"]     # what the model tries to predict

    print("\nFeatures (X) shape:", X.shape)
    print("Target  (y) shape :", y.shape)
    print("\nSample feature rows:")
    print(X.head())

    return X, y, le_city, le_furnish, features


if __name__ == "__main__":
    df = load_and_clean()
    X, y, le_city, le_furnish, features = engineer_features(df)
    print("\nFeature engineering complete!")
