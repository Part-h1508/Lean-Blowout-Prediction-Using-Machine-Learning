import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("\nDEEPER ANALYSIS OF 90.xlsx (LBO CONDITION FILE)")
print("="*80)

df = pd.read_excel('data/90.xlsx', header=None)
n = len(df)
sensor = df[1].values

# Calculate rolling statistics to find where instability starts
window = 500
rolling_mean = pd.Series(sensor).rolling(window=window, center=True).mean().values
rolling_std = pd.Series(sensor).rolling(window=window, center=True).std().values

# Normalize std by mean (coefficient of variation)
cv = rolling_std / (rolling_mean + 1e-10)

print(f"\nFile: 90.xlsx")
print(f"Total rows: {n}")
print(f"Looking for sudden changes in coefficient of variation...")
print(f"\nTop 20 rows with highest variability:")

# Find where variability spikes
high_cv_indices = np.argsort(cv)[-20:][::-1]
high_cv_indices = high_cv_indices[~np.isnan(cv[high_cv_indices])]

for idx in high_cv_indices[:20]:
    print(f"  Row {idx:5d}: CV={cv[idx]:.4f}, Mean={rolling_mean[idx]:.5f}, Std={rolling_std[idx]:.5f}")

print(f"\n" + "="*80)
print("STATISTICS BREAKDOWN")
print("="*80)

# Split the file into quarters
quarters = [n//4, n//2, 3*n//4, n]
quarter_names = ["First Quarter", "Second Quarter", "Third Quarter", "Last Quarter"]

for q, name in zip(quarters[:-1], quarter_names):
    start = q - n//4
    end = q
    segment = sensor[start:end]
    print(f"\n{name} (rows {start:5d} - {end:5d}):")
    print(f"  Mean: {segment.mean():.5f}")
    print(f"  Std:  {segment.std():.5f}")
    print(f"  Min:  {segment.min():.5f}")
    print(f"  Max:  {segment.max():.5f}")
    print(f"  CV:   {segment.std()/segment.mean():.5f}")

# Look at last 500 rows more carefully
print(f"\nLAST 500 ROWS ANALYSIS:")
last_500 = sensor[-500:]
print(f"  Mean: {last_500.mean():.5f}")
print(f"  Std:  {last_500.std():.5f}")
print(f"  Min:  {last_500.min():.5f} (at index {np.argmin(last_500) + n - 500})")
print(f"  Max:  {last_500.max():.5f}")

# Create plot
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Full signal
axes[0].plot(sensor, linewidth=0.5, alpha=0.8)
axes[0].set_title('Full Signal (90.xlsx)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Sensor Value')
axes[0].grid(True, alpha=0.3)

# Coefficient of Variation
axes[1].plot(cv, linewidth=0.8, color='orange', alpha=0.7)
axes[1].set_title('Coefficient of Variation (rolling window=500)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('CV')
axes[1].grid(True, alpha=0.3)

# Last 5000 rows zoomed in
last_5k_start = n - 5000
axes[2].plot(range(last_5k_start, n), sensor[last_5k_start:], linewidth=0.8, color='red', alpha=0.7)
axes[2].set_title(f'Last 5000 Rows (rows {last_5k_start:d} - {n:d})', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Row Index')
axes[2].set_ylabel('Sensor Value')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lbo_analysis.png', dpi=100)
print(f"\n✓ Analysis plot saved to: lbo_analysis.png")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print("""
Based on the analysis, LBO likely occurs in the last ~2000 rows of the file.
Look at the plot 'lbo_analysis.png' to identify where the signal becomes unstable.

If you have the research paper, it should specify:
- Exact row number where LBO occurs
- Lead time (how many rows before actual LBO the warning should start)

Once you identify the row, set in config.py:
    LBO_ROW_INDEX = [row_number]
""")
