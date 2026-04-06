# Step 6: Compare Linear Regression vs Decision Tree
# ----------------------------------------
# Goal: Train a second model (Decision Tree) and compare both side by side.
# This is how real data scientists pick the best model for their problem.

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from step1_load_clean import load_and_clean
from step3_features import engineer_features
from step4_model import train_linear_regression
from step5_evaluate import evaluate_model

os.makedirs("plots", exist_ok=True)


def compare_models():
    df = load_and_clean()
    X, y, _, _, _ = engineer_features(df)
    lr_model, X_train, X_test, y_train, y_test = train_linear_regression(X, y)

    # --- Decision Tree Regressor ---
    # max_depth=6 limits how complex the tree can grow.
    # Without this limit, the tree memorizes training data (overfitting).
    dt_model = DecisionTreeRegressor(max_depth=6, random_state=42)
    dt_model.fit(X_train, y_train)
    print("\nDecision Tree trained!")

    # Evaluate both models
    print("\n--- Linear Regression ---")
    lr_mae, lr_r2 = evaluate_model(lr_model, X_test, y_test, "Linear Regression")

    print("\n--- Decision Tree ---")
    dt_mae, dt_r2 = evaluate_model(dt_model, X_test, y_test, "Decision Tree")

    # --- Comparison Bar Charts ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = ["steelblue", "coral"]
    model_names = ["Linear\nRegression", "Decision\nTree"]

    # R² comparison
    bars0 = axes[0].bar(model_names, [lr_r2, dt_r2], color=colors, edgecolor="white", width=0.5)
    axes[0].set_title("R² Score (higher = better)", fontsize=12)
    axes[0].set_ylim(0, 1.1)
    for bar, val in zip(bars0, [lr_r2, dt_r2]):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.02,
                     f"{val:.3f}", ha="center", fontsize=11)

    # MAE comparison
    bars1 = axes[1].bar(model_names, [lr_mae, dt_mae], color=colors, edgecolor="white", width=0.5)
    axes[1].set_title("MAE — Mean Absolute Error (lower = better)", fontsize=12)
    for bar, val in zip(bars1, [lr_mae, dt_mae]):
        axes[1].text(bar.get_x() + bar.get_width()/2, val + 200,
                     f"₹{val:,.0f}", ha="center", fontsize=10)

    plt.suptitle("Model Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/model_comparison.png")
    plt.show()
    print("Saved: plots/model_comparison.png")

    # Summary
    print("\n=== Final Summary ===")
    print(f"  Linear Regression → R²: {lr_r2:.4f} | MAE: ₹{lr_mae:,.0f}")
    print(f"  Decision Tree     → R²: {dt_r2:.4f} | MAE: ₹{dt_mae:,.0f}")
    winner = "Decision Tree" if dt_r2 > lr_r2 else "Linear Regression"
    print(f"\n  Best model by R²: {winner}")
    print("\n  Interview tip: Decision Trees capture non-linear patterns (e.g.,")
    print("  luxury areas with sudden price jumps). Linear Regression assumes")
    print("  a straight-line relationship — simpler but less flexible.")

    return lr_model, dt_model, X_test, y_test


if __name__ == "__main__":
    compare_models()
