# Smart Rental Price Prediction 🏠

A beginner-friendly Machine Learning project that predicts rental prices and identifies whether a property is overpriced or fairly priced.

## Tech Stack
- **Python** — core language
- **Pandas** — data loading and cleaning
- **NumPy** — numerical computations
- **Matplotlib** — visualizations
- **Scikit-learn** — ML models (Linear Regression, Decision Tree)

## Project Structure
```
smart-rental-price-prediction/
├── data/
│   └── rental_data.csv          ← dataset (40 sample properties)
├── src/
│   ├── step1_load_clean.py      ← load & clean data
│   ├── step2_eda.py             ← exploratory data analysis
│   ├── step3_features.py        ← feature engineering & encoding
│   ├── step4_model.py           ← train Linear Regression
│   ├── step5_evaluate.py        ← evaluate with R² and MAE
│   ├── step6_compare.py         ← compare with Decision Tree
│   └── step7_price_checker.py   ← overpriced/fair price checker
├── notebooks/
│   └── rental_analysis.ipynb   ← all steps in one notebook
├── plots/                       ← charts generated during EDA
├── requirements.txt
└── README.md
```

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run steps in order (from the project root)
```bash
python src/step1_load_clean.py
python src/step2_eda.py
python src/step3_features.py
python src/step4_model.py
python src/step5_evaluate.py
python src/step6_compare.py
python src/step7_price_checker.py
```

### 3. Or open the Jupyter Notebook
```bash
jupyter notebook notebooks/rental_analysis.ipynb
```

## Key Results

| Model             | R² Score | MAE (₹) |
|-------------------|----------|---------|
| Linear Regression | ~0.80    | ~3,500  |
| Decision Tree     | ~0.85    | ~3,000  |

*Results vary slightly based on train/test split.*

## Features Used
| Feature | Description |
|---|---|
| BHK | Number of bedrooms |
| Size | Property size in sq ft |
| City_encoded | City (label encoded) |
| Furnishing_encoded | Furnishing status (label encoded) |
| Price_per_sqft | Rent ÷ Size (engineered feature) |

## Key Concepts (for interviews)

**R² Score**: How much of the rent variation the model explains. 1.0 = perfect, 0.0 = no better than guessing the mean.

**MAE**: Mean Absolute Error — average rupee difference between actual and predicted rent. Easy to explain: "my model is off by ₹X on average."

**Label Encoding**: Converts text categories (city names, furnishing types) to integers so ML algorithms can process them.

**Train/Test Split**: Trains on 80% of data, tests on unseen 20% — simulates real-world performance.

**Overpriced Checker Logic**: If actual rent > predicted rent by more than 15%, flag as overpriced.

## Dataset
The sample dataset (`rental_data.csv`) includes 40 properties across Mumbai, Delhi, Bangalore, Chennai, and Hyderabad with varying BHK, size, and furnishing status.

For a larger real dataset, download from [Kaggle - House Rent Prediction Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset).

## Author
Built as a data analyst portfolio project.
