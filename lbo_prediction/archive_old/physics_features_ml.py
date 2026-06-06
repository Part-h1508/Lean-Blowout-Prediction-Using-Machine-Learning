import pandas as pd
import numpy as np

from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

# ============================================================================
# LOAD DATA
# ============================================================================
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

csv_file = os.path.join(
    BASE_DIR,
    "data",
    "physics_features.csv"
)
print(csv_file)
df = pd.read_csv(csv_file)
X = df[
    [
        "NRMS",
        "Theta",
        "PSDPeak",
        "MeanAutocorr"
    ]
]

y = df["PhiRatio"]

# ============================================================================
# LEAVE ONE OUT
# ============================================================================

loo = LeaveOneOut()

results = []

for train_idx, test_idx in loo.split(X):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)[0]

    actual = y_test.values[0]

    error = abs(prediction - actual)

    percent_error = (
        error / actual
    ) * 100

    air = df.iloc[test_idx[0]]["Air"]

    results.append([
        air,
        actual,
        prediction,
        error,
        percent_error
    ])

# ============================================================================
# RESULTS
# ============================================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Air",
        "Actual",
        "Predicted",
        "Error",
        "Percent_Error"
    ]
)

print("\n")
print(results_df.to_string(index=False))

print("\nAverage Error:")
print(results_df["Percent_Error"].mean())

print("\nFeature Importance:")

final_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

final_model.fit(X, y)

for feature, importance in zip(
    X.columns,
    final_model.feature_importances_
):
    print(
        f"{feature}: {importance:.4f}"
    )