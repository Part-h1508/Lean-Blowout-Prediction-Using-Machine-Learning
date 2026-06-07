"""
LBO (Lean Blowout) Prediction Pipeline
Complete time-series event prediction system for combustion sensor data
"""

import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR_IMPORT = os.path.join(CURRENT_DIR, "data")

if DATA_DIR_IMPORT not in sys.path:
    sys.path.insert(0, DATA_DIR_IMPORT)

from private_config import PHI_RATIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
from xgboost import XGBRegressor
from imblearn.over_sampling import SMOTE
import pickle
import warnings
from config import (
    DATA_DIR, OUTPUT_DIR, MODELS_DIR, PLOTS_DIR, RESULTS_DIR,
    TIME_COL, LBO_ROW_INDEX, COLS_TO_DROP, WARNING_WINDOW,
    TEST_SIZE, RANDOM_STATE, SMOTE_K_NEIGHBORS, XGBOOST_PARAMS,
    PREDICTION_THRESHOLD, PLOT_DPI, PLOT_FIGSIZE
)
from scipy.signal import welch  

warnings.filterwarnings('ignore')

# Change to script directory so relative paths work
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Create output directories
for directory in [MODELS_DIR, PLOTS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

print("=" * 80)
print("LBO PREDICTION PIPELINE")
print("=" * 80)

# ============================================================================
# SECTION A: DATA LOADING AND VALIDATION
# ============================================================================
print("\n[A] Loading and validating data...")

def load_data():
    """Load sensor data files from the data directory."""
    try:
        # Only load numeric-named xlsx files (65.xlsx through 90.xlsx)
        all_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.xlsx')]
        sensor_files = []
        for f in all_files:
            try:
                # Try to parse filename as number (e.g., "65.xlsx" -> 65)
                float(f.replace('.xlsx', ''))
                sensor_files.append(f)
            except ValueError:
                # Skip non-numeric files (like "Data details.xlsx")
                continue
        
        sensor_files.sort(key=lambda x: float(x.replace('.xlsx', '')))
        
        if not sensor_files:
            raise FileNotFoundError(f"No sensor files (numeric names) found in {DATA_DIR}")
        
        print(f"   Found {len(sensor_files)} sensor file(s)")
        
        dataframes = []
        for file in sensor_files:
            filepath = os.path.join(DATA_DIR, file)
            try:
                # Read Excel file without header (all rows are data)
                df = pd.read_excel(filepath, header=None)

                # Store source file
                df["source_file"] = file.replace(".xlsx", "")

                print(f"   Loaded: {file} ({len(df)} rows, {len(df.columns)-1} columns)")

                dataframes.append(df)
            except Exception as e:
                print(f"   WARNING: Could not load {file}: {e}")
        
        if not dataframes:
            raise ValueError("No dataframes could be loaded successfully")
        
        # Concatenate all dataframes
        df = pd.concat(dataframes, ignore_index=True)
        return df
    
    except Exception as e:
        print(f"   ERROR in load_data: {e}")
        sys.exit(1)

df = load_data()
print(f"   Total shape: {df.shape}")
print(f"   Columns: {list(df.columns)[:10]}..." if len(df.columns) > 10 else f"   Columns: {list(df.columns)}")

# ============================================================================
# SECTION B: DATA PREPROCESSING
# ============================================================================
print("\n[B] Preprocessing data...")

def preprocess_data(df):
    """Clean and prepare data."""
    try:
        # Drop specified columns
        cols_to_drop_valid = [col for col in COLS_TO_DROP if col in df.columns]
        if cols_to_drop_valid:
            print(f"   Dropping columns: {cols_to_drop_valid}")
            df = df.drop(columns=cols_to_drop_valid)
        
        # Drop TIME_COL if specified and present
        if TIME_COL and TIME_COL in df.columns and TIME_COL != "":
            print(f"   Dropping TIME_COL: {TIME_COL}")
            df = df.drop(columns=[TIME_COL])
        
        # Remove rows with NaN
        initial_rows = len(df)
        df = df.dropna()
        rows_dropped = initial_rows - len(df)
        if rows_dropped > 0:
            print(f"   Dropped {rows_dropped} rows with NaN values")
        
        # Ensure all columns are numeric
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()
        
        print(f"   Final shape after preprocessing: {df.shape}")
        return df
    
    except Exception as e:
        print(f"   ERROR in preprocess_data: {e}")
        sys.exit(1)

df = preprocess_data(df)
source_files = df["source_file"].copy()
# Normalize source file identifiers to string keys (match PHI_RATIO keys)
source_files = source_files.astype(str).str.strip()
df = df.drop(columns=["source_file"])

"""
# ============================================================================
# SECTION C: VALIDATE LBO_ROW_INDEX
# ============================================================================
print("\n[C] Validating LBO_ROW_INDEX...")

if LBO_ROW_INDEX is None:
    print("   ERROR: LBO_ROW_INDEX is not set in config.py")
    print("   ACTION: Open config.py, inspect your data, find where LBO occurs, and set LBO_ROW_INDEX")
    print("   Exiting...")
    sys.exit(1)

try:
    if LBO_ROW_INDEX < 0 or LBO_ROW_INDEX >= len(df):
        print(f"   ERROR: LBO_ROW_INDEX ({LBO_ROW_INDEX}) is out of bounds")
        print(f"   Valid range: 0 to {len(df) - 1}")
        sys.exit(1)
    print(f"   LBO_ROW_INDEX: {LBO_ROW_INDEX} (valid)")
except Exception as e:
    print(f"   ERROR validating LBO_ROW_INDEX: {e}")
    sys.exit(1)
"""

# ============================================================================
# SECTION D: FEATURE ENGINEERING
# ============================================================================
print("\n[D] Feature engineering...")

def create_features(df):
    """Create time-series features from sensor data."""
    try:
        df_features = df.copy()
        
        # Rolling statistics (window of 10, 20, 50 rows)
        for window in [50, 100, 500]:
            for col in df.columns:
                df_features[f'{col}_roll_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                df_features[f'{col}_roll_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()

        # NRMS features
        for col in df.columns:

            rolling_mean = df[col].rolling(window=500, min_periods=1).mean()
            rolling_std = df[col].rolling(window=500, min_periods=1).std()

            df_features[f'{col}_nrms'] = rolling_std / (rolling_mean.abs() + 1e-8)
        
        # Lagged features (previous 1, 2, 5 rows)
        for lag in [1, 2, 5]:
            for col in df.columns:
                df_features[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Differences (rate of change)
        for col in df.columns:
            df_features[f'{col}_diff'] = df[col].diff()
        # ============================================================
        # PHYSICS FEATURES
        # ============================================================

        for col in df.columns:

            # --------------------------
            # Theta
            # --------------------------

            rolling_mean = df[col].rolling(
                window=500,
                min_periods=1
            ).mean()

            threshold = 0.72 * rolling_mean

            theta = (
                df[col] < threshold
            ).rolling(
                window=500,
                min_periods=1
            ).mean()

            df_features[f'{col}_theta'] = theta


            # --------------------------
            # Autocorrelation
            # --------------------------

            df_features[f'{col}_autocorr_1'] = (
                df[col]
                .rolling(window=500, min_periods=50)
                .corr(df[col].shift(1))
            )


            # --------------------------
            # PSD Energy Proxy
            # --------------------------

            df_features[f'{col}_psd_proxy'] = (
                df[col]
                .rolling(window=500, min_periods=1)
                .std()
                ** 2
            )
        
        # Fill NaN values created by rolling/lagging
        df_features = df_features.bfill().ffill().fillna(0)
        
        print(f"   Created {len(df_features.columns)} features from {len(df.columns)} original columns")
        print(f"   Features shape: {df_features.shape}")
        print("\nCreated Features:")
        print(df_features.columns.tolist())
        print(f"\nTotal Features: {len(df_features.columns)}")
        return df_features
    
    except Exception as e:
        print(f"   ERROR in create_features: {e}")
        sys.exit(1)

df_features = create_features(df)

# Convert all column names to strings for sklearn compatibility
df_features.columns = df_features.columns.astype(str)
# ============================================================================
# SECTION E: CREATE REGRESSION TARGET
# ============================================================================
print("\n[E] Creating Phi/Phi_LBO targets...")

try:
    # Convert source_files to strings to match PHI_RATIO keys
    source_files_str = source_files.astype(str)
    
    y = source_files_str.map(PHI_RATIO)

    if y.isna().any():
        print("ERROR: Some source files were not found in PHI_RATIO dictionary")
        print(f"   NaN values found at {y.isna().sum()} locations")
        sys.exit(1)

    print(f"   Created {len(y)} target values")
    print(f"   Target range: {y.min():.4f} to {y.max():.4f}")

except Exception as e:
    print(f"   ERROR creating targets: {e}")
    sys.exit(1)

# ============================================================================
# SECTION F: TRAIN/TEST SPLIT (FILE-BASED)
# ============================================================================
print("\n[F] File-based train/test split...")

test_file = "88" # important as this can be used in leave one out approach

train_mask = source_files != test_file
test_mask = source_files == test_file

X_train = df_features[train_mask]
X_test = df_features[test_mask]

y_train = y[train_mask]
y_test = y[test_mask]

print(f"   Training files: 65.xlsx - 89.xlsx")
print(f"   Test file: {test_file}.xlsx")

print(f"   Train set: {len(X_train)} rows")
print(f"   Test set: {len(X_test)} rows")

print(f"   Train target range: {y_train.min():.4f} to {y_train.max():.4f}")
print(f"   Test target value: {y_test.iloc[0]:.4f}")

# ============================================================================
# SECTION G: FEATURE SCALING
# ============================================================================
print("\n[G] Scaling features...")

def scale_features(X_train, X_test):
    """Scale features using StandardScaler."""
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"   Features scaled (StandardScaler)")
        return X_train_scaled, X_test_scaled, scaler
    
    except Exception as e:
        print(f"   ERROR in scale_features: {e}")
        sys.exit(1)

X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

# ============================================================================
# SECTION H: CLASS BALANCING WITH SMOTE
# ============================================================================
print("\n[H] Preparing training data for regression...")

# For regression, SMOTE is not applicable (no class balancing needed)
print("   Regression task - skipping SMOTE")
X_train_balanced = X_train_scaled
y_train_balanced = y_train
smote = None

# ============================================================================
# SECTION I: MODEL TRAINING
# ============================================================================
print("\n[I] Training XGBoost model...")

def train_model(X_train, y_train):
    """Train XGBoost regressor."""
    try:
        model = XGBRegressor(**XGBOOST_PARAMS)
        model.fit(X_train, y_train)
        
        print(f"   Model trained successfully")
        print(f"   Training set size: {len(X_train)}")
        
        return model
    
    except Exception as e:
        print(f"   ERROR in train_model: {e}")
        sys.exit(1)

model = train_model(X_train_balanced, y_train_balanced)

# ============================================================================
# SECTION J: EVALUATION AND METRICS
# ============================================================================
print("\n[J] Evaluating model...")

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluate regression model on train and test sets."""
    try:
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Regression Metrics
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
        
        print(f"   Training MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
        print(f"   Test MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
        print(f"   Test MAPE: {test_mape:.4f}")
        
        return y_train_pred, y_test_pred, test_mae, test_rmse, test_r2
    
    except Exception as e:
        print(f"   ERROR in evaluate_model: {e}")
        sys.exit(1)

y_train_pred, y_test_pred, test_mae, test_rmse, test_r2 = evaluate_model(
    model, X_train_scaled, X_test_scaled, y_train, y_test
)

# ============================================================================
# SAVING MODEL AND PREPROCESSING OBJECTS
# ============================================================================
print("\n[K] Saving model and preprocessing objects...")

try:
    model_path = os.path.join(MODELS_DIR, 'lbo_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"   Model saved: {model_path}")
    
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"   Scaler saved: {scaler_path}")
except Exception as e:
    print(f"   ERROR saving model: {e}")

# ============================================================================
# PLOTTING
# ============================================================================
print("\n[L] Generating plots...")

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Plot 1: Actual vs Predicted
try:
    print("   Generating: Actual vs Predicted plot...")
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    
    ax.scatter(y_test, y_test_pred, alpha=0.5, s=10)
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Phi/Phi_LBO', fontsize=11)
    ax.set_ylabel('Predicted Phi/Phi_LBO', fontsize=11)
    ax.set_title('Actual vs Predicted (Regression)', fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, 'actual_vs_predicted.png')
    plt.savefig(plot_path, dpi=PLOT_DPI)
    plt.close()
    print(f"      Saved: {plot_path}")
except Exception as e:
    print(f"      ERROR: {e}")

# Plot 2: Residuals
try:
    print("   Generating: Residuals plot...")
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    
    residuals = y_test - y_test_pred
    ax.scatter(y_test_pred, residuals, alpha=0.5, s=10)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    
    ax.set_xlabel('Predicted Phi/Phi_LBO', fontsize=11)
    ax.set_ylabel('Residuals', fontsize=11)
    ax.set_title('Residuals Plot', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, 'residuals.png')
    plt.savefig(plot_path, dpi=PLOT_DPI)
    plt.close()
    print(f"      Saved: {plot_path}")
except Exception as e:
    print(f"      ERROR: {e}")

# Plot 3: Feature Importance
try:
    print("   Generating: Feature importance...")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=PLOT_DPI)
    feature_importance = model.feature_importances_
    top_indices = np.argsort(feature_importance)[-15:]  # Top 15 features
    top_features = [df_features.columns[i] for i in top_indices]
    top_importance = feature_importance[top_indices]
    
    ax.barh(range(len(top_features)), top_importance)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel('Importance', fontsize=11)
    ax.set_title('Top 15 Feature Importances', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fi_path = os.path.join(PLOTS_DIR, 'feature_importance.png')
    plt.savefig(fi_path, dpi=PLOT_DPI)
    plt.close()
    print(f"      Saved: {fi_path}")
except Exception as e:
    print(f"      ERROR: {e}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    pr_path = os.path.join(PLOTS_DIR, 'precision_recall_curve.png')
    plt.savefig(pr_path, dpi=PLOT_DPI)
    plt.close()
    print(f"      Saved: {pr_path}")
except Exception as e:
    print(f"      ERROR: {e}")

# No precision-recall curve for regression - removed classification plot

# ============================================================================
# GENERATING RESULTS REPORT
# ============================================================================
print("\n[M] Generating results report...")

try:
    report_path = os.path.join(RESULTS_DIR, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("LBO PREDICTION PIPELINE - RESULTS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("DATA SUMMARY:\n")
        f.write(f"  Total samples: {len(df)}\n")
        f.write(f"  Total features (after engineering): {len(df_features.columns)}\n")
        f.write(f"  LBO row index: {LBO_ROW_INDEX}\n")
        f.write(f"  Warning window: {WARNING_WINDOW} rows\n")
        f.write(f"  Total LBO samples: {int(y.sum())}\n\n")
        
        f.write("TRAIN/TEST SPLIT:\n")
        f.write(f"  Training samples: {len(X_train)}\n")
        f.write(f"  Test samples: {len(X_test)}\n")
        f.write(f"  Training LBO samples: {int(y_train.sum())}\n")
        f.write(f"  Test LBO samples: {int(y_test.sum())}\n\n")
        
        f.write("MODEL PERFORMANCE:\n")
        f.write(f"  Test MAE: {test_mae:.4f}\n")
        f.write(f"  Test RMSE: {test_rmse:.4f}\n")
        f.write(f"  Test R²: {test_r2:.4f}\n")
        f.write(f"  Mean Absolute Percentage Error: {mean_absolute_percentage_error(y_test, y_test_pred):.4f}\n\n")
        
        f.write(f"  Test MAPE: {mean_absolute_percentage_error(y_test, y_test_pred):.4f}\n\n")
        
        f.write("PREDICTION SAMPLES:\n")
        f.write(f"  First 10 predictions vs actual:\n")
        for i in range(min(10, len(y_test))):
            f.write(f"    Actual: {y_test.iloc[i]:.4f}, Predicted: {y_test_pred[i]:.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("FILES GENERATED:\n")
        f.write("  Models:\n")
        f.write(f"    - {model_path}\n")
        f.write(f"    - {scaler_path}\n")
        f.write("  Plots:\n")
        f.write(f"    - {os.path.join(PLOTS_DIR, 'actual_vs_predicted.png')}\n")
        f.write(f"    - {os.path.join(PLOTS_DIR, 'residuals.png')}\n")
        f.write(f"    - {os.path.join(PLOTS_DIR, 'feature_importance.png')}\n")
        f.write(f"    - {report_path}\n")
    
    print(f"   Report saved: {report_path}")
except Exception as e:
    print(f"   ERROR generating report: {e}")

# ============================================================================
# FINAL INSTRUCTIONS
# ============================================================================
print("\n[N] Creating instructions file...")

try:
    instructions_path = os.path.join(RESULTS_DIR, 'INSTRUCTIONS.txt')
    with open(instructions_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("LBO PREDICTION PIPELINE - INSTRUCTIONS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("NEXT STEPS:\n\n")
        
        f.write("1. REVIEW THE TIMELINE PLOT:\n")
        f.write("   Open: outputs/plots/timeline.png\n")
        f.write("   - This is your paper's key figure\n")
        f.write("   - Shows predicted LBO probability rising BEFORE the actual event\n")
        f.write("   - Demonstrates the predictive power of the model\n\n")
        
        f.write("2. IF PREDICTION IS TOO EARLY/LATE:\n")
        f.write("   - Open: config.py\n")
        f.write("   - Adjust: WARNING_WINDOW parameter\n")
        f.write("   - Increase WARNING_WINDOW to warn earlier\n")
        f.write("   - Decrease WARNING_WINDOW to warn later\n")
        f.write("   - Re-run: python pipeline.py\n\n")
        
        f.write("3. CHECK OTHER PLOTS:\n")
        f.write("   - confusion_matrix.png: Model accuracy breakdown\n")
        f.write("   - roc_curve.png: Model discrimination ability (AUC-ROC)\n")
        f.write("   - feature_importance.png: Top sensors driving LBO prediction\n")
        f.write("   - precision_recall_curve.png: Precision vs Recall tradeoff\n\n")
        
        f.write("4. REVIEW METRICS:\n")
        f.write("   Open: outputs/results/classification_report.txt\n")
        f.write("   - Precision: Of predicted LBO events, how many were correct?\n")
        f.write("   - Recall: Of actual LBO events, how many were caught?\n")
        f.write("   - F1: Harmonic mean of precision and recall\n")
        f.write("   - AUC-ROC: Area under the ROC curve (0.5 = random, 1.0 = perfect)\n\n")
        
        f.write("5. USE THE MODEL FOR OTHER EVENTS:\n")
        f.write("   The same pipeline structure works for RBO and TAI\n")
        f.write("   - Create new config files: config_rbo.py, config_tai.py\n")
        f.write("   - Set different LBO_ROW_INDEX values for each event\n")
        f.write("   - Or modify pipeline.py to accept event type as parameter\n\n")
        
        f.write("6. DEPLOYMENT:\n")
        f.write("   - Saved model: outputs/models/lbo_model.pkl\n")
        f.write("   - Saved scaler: outputs/models/scaler.pkl\n")
        f.write("   - Load in production to make real-time predictions\n\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"   Instructions saved: {instructions_path}")
except Exception as e:
    print(f"   ERROR creating instructions: {e}")

# ============================================================================
# PIPELINE COMPLETE
# ============================================================================
print("\n" + "=" * 80)
print("PIPELINE COMPLETE!")
print("=" * 80)
print(f"\nAll outputs saved to: {os.path.abspath(OUTPUT_DIR)}")
print(f"\nKey files:")
print(f"  - Timeline plot: {os.path.join(PLOTS_DIR, 'timeline.png')}")
print(f"  - Model metrics: {os.path.join(RESULTS_DIR, 'classification_report.txt')}")
print(f"  - Instructions: {os.path.join(RESULTS_DIR, 'INSTRUCTIONS.txt')}")
print(f"\nNext: Adjust WARNING_WINDOW in config.py and re-run if needed.")
print("=" * 80 + "\n")
