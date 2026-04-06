# Step 4: Train a Linear Regression Model
# ----------------------------------------
# Goal: Train a model that learns the relationship between
# property features and rent price from historical data.

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from step1_load_clean import load_and_clean
from step3_features import engineer_features


def train_linear_regression(X, y):
    # Split data: 80% for training, 20% for testing
    # random_state=42 → same split every time you run (reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training samples : {len(X_train)}")
    print(f"Testing  samples : {len(X_test)}")

    # Create and train the Linear Regression model
    # "fit" = the model learns from training data
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Show what the model learned
    print("\n=== Model Coefficients ===")
    print("(Each value = how much rent changes per unit increase in that feature)")
    for feature, coef in zip(X.columns, model.coef_):
        direction = "↑ increases rent" if coef > 0 else "↓ decreases rent"
        print(f"  {feature:30s}: {coef:>10,.2f}  {direction}")
    print(f"  {'Intercept (base rent)':30s}: {model.intercept_:>10,.2f}")

    return model, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = load_and_clean()
    X, y, _, _, _ = engineer_features(df)
    model, X_train, X_test, y_train, y_test = train_linear_regression(X, y)
    print("\nLinear Regression model trained successfully!")
