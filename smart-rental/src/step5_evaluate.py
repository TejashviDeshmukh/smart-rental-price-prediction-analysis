# Step 5: Evaluate Model Performance
# ----------------------------------------
# Goal: Measure how accurate our model is.
# Two key metrics:
#   R² (R-squared) — how much of the price variation does the model explain?
#   MAE (Mean Absolute Error) — on average, how many rupees off are we?

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from step1_load_clean import load_and_clean
from step3_features import engineer_features
from step4_model import train_linear_regression

os.makedirs("plots", exist_ok=True)


def evaluate_model(model, X_test, y_test, model_name="Linear Regression"):
    # Make predictions on the test set (data the model never saw)
    y_pred = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*50}")
    print(f"  R² Score : {r2:.4f}")
    print(f"             (1.0 = perfect | 0.0 = no better than guessing the average)")
    print(f"  MAE      : ₹{mae:,.0f}")
    print(f"             (average prediction error in rupees)")
    print(f"{'='*50}")

    # Actual vs Predicted scatter plot
    # If the model is perfect, all dots fall on the red diagonal line
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, y_pred, alpha=0.6, color="steelblue", edgecolors="none")
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--", linewidth=1.5, label="Perfect prediction"
    )
    plt.xlabel("Actual Rent (₹)")
    plt.ylabel("Predicted Rent (₹)")
    plt.title(f"Actual vs Predicted — {model_name}")
    plt.legend()
    plt.tight_layout()
    fname = f"plots/actual_vs_predicted_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(fname)
    plt.show()
    print(f"Saved: {fname}")

    return mae, r2


if __name__ == "__main__":
    df = load_and_clean()
    X, y, _, _, _ = engineer_features(df)
    model, X_train, X_test, y_train, y_test = train_linear_regression(X, y)
    evaluate_model(model, X_test, y_test)
