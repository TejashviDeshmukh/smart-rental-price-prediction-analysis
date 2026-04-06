# Step 7: Overpriced / Fairly Priced Checker
# ----------------------------------------
# Goal: Given a real property listing, predict its fair rent and
# tell the user whether it's overpriced, a great deal, or fairly priced.
# This is the most practical output of the whole project!

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from step1_load_clean import load_and_clean
from step3_features import engineer_features
from step4_model import train_linear_regression


def build_checker():
    """
    Trains the model and returns a ready-to-use checker function.
    Call this once, then use the returned function many times.
    """
    df = load_and_clean()
    X, y, le_city, le_furnish, features = engineer_features(df)
    model, X_train, X_test, y_train, y_test = train_linear_regression(X, y)

    print("\nChecker ready! Valid cities:", list(le_city.classes_))
    print("Valid furnishing:", list(le_furnish.classes_))

    def check_pricing(size, bhk, city_name, furnishing_status, actual_rent):
        """
        Parameters:
          size             : property size in sq ft (e.g., 1200)
          bhk              : number of bedrooms (e.g., 2)
          city_name        : must match a city in the dataset (e.g., "Mumbai")
          furnishing_status: "Furnished", "Semi-Furnished", or "Unfurnished"
          actual_rent      : the listed rent in rupees (e.g., 35000)
        """
        # Encode inputs using the SAME encoders used during training
        try:
            city_enc    = le_city.transform([city_name])[0]
            furnish_enc = le_furnish.transform([furnishing_status])[0]
        except ValueError as e:
            print(f"\nError: {e}")
            print(f"Valid cities    : {list(le_city.classes_)}")
            print(f"Valid furnishing: {list(le_furnish.classes_)}")
            return

        price_per_sqft = actual_rent / size

        # Build input as a DataFrame (same format as training data)
        input_df = pd.DataFrame(
            [[bhk, size, city_enc, furnish_enc, price_per_sqft]],
            columns=features
        )

        # Predict the fair rent
        predicted_rent = model.predict(input_df)[0]

        # Compare actual vs predicted
        difference   = actual_rent - predicted_rent
        percent_diff = (difference / predicted_rent) * 100

        # Print the report
        print("\n" + "=" * 48)
        print("       RENTAL PRICE ANALYSIS REPORT")
        print("=" * 48)
        print(f"  City              : {city_name}")
        print(f"  BHK               : {bhk}")
        print(f"  Size              : {size:,} sq ft")
        print(f"  Furnishing        : {furnishing_status}")
        print("-" * 48)
        print(f"  Actual Rent       : ₹{actual_rent:,}")
        print(f"  Predicted Rent    : ₹{predicted_rent:,.0f}")
        print(f"  Difference        : ₹{difference:+,.0f}  ({percent_diff:+.1f}%)")
        print("-" * 48)

        if percent_diff > 15:
            print("  Verdict  :  ⚠️  OVERPRICED")
            print(f"              You'd pay ₹{abs(difference):,.0f} MORE than fair value")
            print("              Tip: Negotiate or keep looking!")
        elif percent_diff < -15:
            print("  Verdict  :  ✅  UNDERPRICED — Great Deal!")
            print(f"              You'd SAVE ₹{abs(difference):,.0f} vs fair market value")
            print("              Tip: Move fast — this won't last!")
        else:
            print("  Verdict  :  ✅  FAIRLY PRICED")
            print("              Within 15% of predicted fair market rent")
        print("=" * 48)

    return check_pricing


if __name__ == "__main__":
    checker = build_checker()

    print("\n\n--- Test 1: Likely Overpriced (high rent for the area) ---")
    checker(size=900, bhk=2, city_name="Delhi",
            furnishing_status="Furnished", actual_rent=45000)

    print("\n\n--- Test 2: Fairly Priced ---")
    checker(size=1200, bhk=2, city_name="Mumbai",
            furnishing_status="Semi-Furnished", actual_rent=27000)

    print("\n\n--- Test 3: Potential Deal ---")
    checker(size=1500, bhk=3, city_name="Bangalore",
            furnishing_status="Furnished", actual_rent=40000)

    print("\n\n--- Test 4: Try your own property ---")
    # Modify these values and re-run to check any property!
    checker(
        size=1000,
        bhk=2,
        city_name="Chennai",
        furnishing_status="Unfurnished",
        actual_rent=20000
    )
