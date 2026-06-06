import pandas as pd
import numpy as np
import os

print("\n" + "="*80)
print("ANALYZING ALL DATA FILES FOR LBO EVENT")
print("="*80)

# Get all xlsx files
data_dir = "data/"
xlsx_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.xlsx') and f != 'Data details.xlsx'])

print(f"\nFound {len(xlsx_files)} data files\n")

results = []

for file in xlsx_files:
    filepath = os.path.join(data_dir, file)
    df = pd.read_excel(filepath, header=None)
    
    sensor_col = 1  # Second column (index 1)
    
    # Get basic stats
    n_rows = len(df)
    mean = df[sensor_col].mean()
    std = df[sensor_col].std()
    
    # Look at last 20% to detect LBO
    last_20pct_start = int(0.8 * n_rows)
    last_20pct = df[sensor_col].iloc[last_20pct_start:]
    
    # Check if there's a significant deviation in the last 20%
    last_20pct_mean = last_20pct.mean()
    ratio = last_20pct_mean / mean if mean != 0 else 1.0
    
    # Calculate variance to detect changes
    changes = df[sensor_col].diff().abs()
    big_drops = changes.nsmallest(5)
    
    results.append({
        'file': file,
        'rows': n_rows,
        'mean': mean,
        'std': std,
        'last_20pct_mean': last_20pct_mean,
        'ratio': ratio,
        'min_val': df[sensor_col].min(),
        'max_val': df[sensor_col].max()
    })
    
    print(f"{file:15} | Rows: {n_rows:5d} | Mean: {mean:.5f} | Last 20% Mean: {last_20pct_mean:.5f} | Ratio: {ratio:.3f}")
    print(f"                | Min: {df[sensor_col].min():.5f} | Max: {df[sensor_col].max():.5f}")
    print()

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\nFiles are organized by Air flow rate (SLPM):")
print("65, 70, 75, 80, 85, 86, 87, 88, 89, 90 SLPM")
print("\nAccording to your data details:")
print("- 90 SLPM = LBO condition (Fi/FI_LBO = 1.0)")
print("- Below 90 SLPM = Supercritical conditions (before LBO)")
print("\nFor LBO model, use 90.xlsx (the file at LBO condition)")
print("\nYOU NEED TO:")
print("1. Open 90.xlsx and visually inspect where LBO occurs")
print("2. Look for when the combustion becomes unstable (sensor value drops sharply)")
print("3. Note the row number where instability starts")
print("4. Set that row number as LBO_ROW_INDEX in config.py")
print("\nTYPICAL LBO CHARACTERISTICS:")
print("- Sudden drop in sensor values")
print("- High variance/oscillation")
print("- Could be near the end (~row 35000-39500 in a 40000-row file)")
print("\n" + "="*80)
