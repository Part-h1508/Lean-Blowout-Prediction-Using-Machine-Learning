import os
import sys
import pandas as pd
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR_IMPORT = os.path.join(CURRENT_DIR, "data")

if DATA_DIR_IMPORT not in sys.path:
    sys.path.insert(0, DATA_DIR_IMPORT)


from private_config import PHI_RATIO

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# ============================================================================
# CONFIG
# ============================================================================

DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "data"
)

XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "objective": "reg:squarederror"
}

# ============================================================================
# SECTION A: DATA LOADING AND VALIDATION
# ============================================================================

print("\n[A] Loading and validating data...")

def load_data():
    """Load sensor data files from the data directory."""

    try:

        all_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.xlsx')]

        sensor_files = []

        for f in all_files:

            try:
                float(f.replace('.xlsx', ''))
                sensor_files.append(f)

            except ValueError:
                continue

        sensor_files.sort(key=lambda x: float(x.replace('.xlsx', '')))

        if not sensor_files:
            raise FileNotFoundError(
                f"No sensor files found in {DATA_DIR}"
            )

        print(f"   Found {len(sensor_files)} sensor file(s)")

        dataframes = []

        for file in sensor_files:

            filepath = os.path.join(DATA_DIR, file)

            try:

                df = pd.read_excel(filepath, header=None)

                df["source_file"] = file.replace(".xlsx", "")

                print(
                    f"   Loaded: {file} "
                    f"({len(df)} rows, {len(df.columns)-1} columns)"
                )

                dataframes.append(df)

            except Exception as e:

                print(
                    f"   WARNING: Could not load {file}: {e}"
                )

        if not dataframes:
            raise ValueError(
                "No dataframes could be loaded successfully"
            )

        df = pd.concat(dataframes, ignore_index=True)

        return df

    except Exception as e:

        print(f"   ERROR in load_data: {e}")
        sys.exit(1)

df = load_data()

print(f"   Total shape: {df.shape}")
print(f"   Columns: {list(df.columns)}")

# ============================================================================
# PREPROCESSING
# ============================================================================

print("\n[B] Preprocessing data...")

source_files = df["source_file"].copy()

df = df.drop(columns=["source_file"])

df = df.apply(pd.to_numeric, errors="coerce")

df = df.dropna()

print(f"   Shape after preprocessing: {df.shape}")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

print("\n[C] Feature engineering...")

df_features = df.copy()

for window in [50, 100, 500]:

    for col in df.columns:

        df_features[f"{col}_roll_mean_{window}"] = (
            df[col]
            .rolling(window=window, min_periods=1)
            .mean()
        )

        df_features[f"{col}_roll_std_{window}"] = (
            df[col]
            .rolling(window=window, min_periods=1)
            .std()
        )

for col in df.columns:

    rolling_mean = (
        df[col]
        .rolling(window=500, min_periods=1)
        .mean()
    )

    rolling_std = (
        df[col]
        .rolling(window=500, min_periods=1)
        .std()
    )

    df_features[f"{col}_nrms"] = (
        rolling_std /
        (rolling_mean.abs() + 1e-8)
    )

for lag in [1, 2, 5]:

    for col in df.columns:

        df_features[f"{col}_lag_{lag}"] = (
            df[col].shift(lag)
        )

for col in df.columns:

    df_features[f"{col}_diff"] = (
        df[col].diff()
    )
# ============================================================
# PHYSICS FEATURES
# ============================================================

for col in df.columns:

    # --------------------------
    # Theta
    # --------------------------

    rolling_mean = (
        df[col]
        .rolling(window=500, min_periods=1)
        .mean()
    )

    threshold = 0.72 * rolling_mean

    theta = (
        (df[col] < threshold)
        .rolling(window=500, min_periods=1)
        .mean()
    )

    df_features[f"{col}_theta"] = theta

    # --------------------------
    # Autocorrelation
    # --------------------------

    df_features[f"{col}_autocorr_1"] = (
        df[col]
        .rolling(window=500, min_periods=50)
        .corr(df[col].shift(1))
    )

    # --------------------------
    # PSD Proxy
    # --------------------------

    df_features[f"{col}_psd_proxy"] = (
        df[col]
        .rolling(window=500, min_periods=1)
        .std()
        ** 2
    )

df_features = (
    df_features
    .bfill()
    .ffill()
    .fillna(0)
)

df_features.columns = (
    df_features.columns.astype(str)
)

print(f"   Feature shape: {df_features.shape}")

# ============================================================================
# TARGETS
# ============================================================================

print("\n[D] Creating targets...")

y = source_files.map(PHI_RATIO)

print(
    f"   Target range: "
    f"{y.min():.4f} to {y.max():.4f}"
)
# ============================================================================
# LEAVE ONE FILE OUT TEST
# ============================================================================

print("\n[E] Running leave-one-file-out validation...")

results = []

def classify(phi):

    if phi > 1.05:
        return "Safe"

    elif phi > 1.025:
        return "Warning"

    else:
        return "Critical"

for test_file in PHI_RATIO.keys():

    print(f"\n   Testing {test_file}.xlsx")

    train_mask = source_files != test_file
    test_mask = source_files == test_file

    X_train = df_features[train_mask]
    X_test = df_features[test_mask]

    y_train = y[train_mask]
    y_test = y[test_mask]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = XGBRegressor(**XGBOOST_PARAMS)

    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    actual_value = float(y_test.iloc[0])

    predicted_mean = float(
        np.mean(predictions)
    )

    percent_error = (
        abs(predicted_mean - actual_value)
        / actual_value
        * 100
    )

    actual_state = classify(actual_value)

    predicted_state = classify(predicted_mean)

    results.append([
        test_file,
        actual_value,
        predicted_mean,
        actual_state,
        predicted_state,
        mae,
        percent_error
    ])
# ============================================================================
# RESULTS TABLE
# ============================================================================

print("\n[F] Results")

results_df = pd.DataFrame(
    results,
    columns=[
        "File",
        "Actual_PhiRatio",
        "Predicted_PhiRatio",
        "Actual_State",
        "Predicted_State",
        "MAE",
        "Percent_Error"
    ]
)

print("\n")
print(results_df.to_string(index=False))

print("\nAverage MAE:")
print(results_df["MAE"].mean())

print("\nAverage Percent Error:")
print(f"{results_df['Percent_Error'].mean():.2f}%")

results_df.to_csv(
    "leave_one_out_results.csv",
    index=False
)

print("\nSaved:")
print("leave_one_out_results.csv")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ============================================================================
# CONFUSION MATRIX
# ============================================================================

actual_states = results_df["Actual_State"]
predicted_states = results_df["Predicted_State"]

labels = ["Safe", "Warning", "Critical"]

cm = confusion_matrix(
    actual_states,
    predicted_states,
    labels=labels
)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

disp.plot(values_format="d")

plt.title("LBO Warning State Classification")
plt.tight_layout()

plt.savefig(
    "confusion_matrix_states.png",
    dpi=300
)

print("\nSaved:")
print("confusion_matrix_states.png")

plt.show()