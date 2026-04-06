# Step 2: Exploratory Data Analysis (EDA)
# ----------------------------------------
# Goal: Understand the data visually before building any model.
# Spot patterns, outliers, and relationships between columns.

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from step1_load_clean import load_and_clean

os.makedirs("plots", exist_ok=True)


def run_eda(df):
    # --- Plot 1: Distribution of Rent Prices ---
    # Shows how rent values are spread — is it skewed? Any outliers?
    plt.figure(figsize=(8, 4))
    plt.hist(df["Rent"], bins=30, color="steelblue", edgecolor="white")
    plt.title("Distribution of Rent Prices")
    plt.xlabel("Rent (₹)")
    plt.ylabel("Number of Properties")
    plt.tight_layout()
    plt.savefig("plots/rent_distribution.png")
    plt.show()
    print("Saved: plots/rent_distribution.png")

    # --- Plot 2: Average Rent by City ---
    # Which city has the highest average rent?
    avg_city = df.groupby("City")["Rent"].mean().sort_values(ascending=False)
    plt.figure(figsize=(8, 4))
    avg_city.plot(kind="bar", color="coral", edgecolor="white")
    plt.title("Average Rent by City")
    plt.ylabel("Average Rent (₹)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("plots/rent_by_city.png")
    plt.show()
    print("Saved: plots/rent_by_city.png")

    # --- Plot 3: Rent vs Size Scatter Plot ---
    # Do larger properties cost more? This shows the relationship.
    plt.figure(figsize=(7, 5))
    plt.scatter(df["Size"], df["Rent"], alpha=0.5, color="teal", edgecolors="none")
    plt.title("Rent vs Property Size")
    plt.xlabel("Size (sq ft)")
    plt.ylabel("Rent (₹)")
    plt.tight_layout()
    plt.savefig("plots/rent_vs_size.png")
    plt.show()
    print("Saved: plots/rent_vs_size.png")

    # --- Plot 4: Average Rent by Furnishing Status ---
    avg_furnish = df.groupby("Furnishing Status")["Rent"].mean().sort_values(ascending=False)
    plt.figure(figsize=(6, 4))
    avg_furnish.plot(kind="bar", color="mediumpurple", edgecolor="white")
    plt.title("Average Rent by Furnishing Status")
    plt.ylabel("Average Rent (₹)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("plots/rent_by_furnishing.png")
    plt.show()
    print("Saved: plots/rent_by_furnishing.png")

    # --- Plot 5: Correlation Matrix ---
    # Shows which numeric columns are related to each other.
    # Values close to 1 or -1 = strong relationship.
    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr()
    plt.figure(figsize=(5, 4))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(corr)), corr.columns, rotation=45)
    plt.yticks(range(len(corr)), corr.columns)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig("plots/correlation.png")
    plt.show()
    print("Saved: plots/correlation.png")

    # Print key stats
    print("\n=== Key Stats ===")
    print(f"Total properties  : {len(df)}")
    print(f"Avg rent          : ₹{df['Rent'].mean():,.0f}")
    print(f"Median rent       : ₹{df['Rent'].median():,.0f}")
    print(f"Min rent          : ₹{df['Rent'].min():,}")
    print(f"Max rent          : ₹{df['Rent'].max():,}")


if __name__ == "__main__":
    df = load_and_clean()
    run_eda(df)
