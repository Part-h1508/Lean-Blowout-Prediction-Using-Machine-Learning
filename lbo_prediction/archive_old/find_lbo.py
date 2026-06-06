import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the LBO condition file (90.xlsx)
df = pd.read_excel('data/90.xlsx', header=None)
print(f"File shape: {df.shape}")

# Plot the sensor data
plt.figure(figsize=(14, 6))
plt.plot(df[1].values, linewidth=0.8, label='Sensor Reading')
plt.xlabel('Row Index')
plt.ylabel('Sensor Reading')
plt.title('Sensor Data (90.xlsx - LBO Condition)\nLook for sharp drop/change = LBO event')
plt.grid(True, alpha=0.3)

# Mark the last 10% where LBO likely occurs
n = len(df)
plt.axvline(x=int(0.90*n), color='red', linestyle='--', alpha=0.5, linewidth=2, label='90% mark')
plt.legend()
plt.tight_layout()
plt.savefig('sensor_plot.png', dpi=100)
print("✓ Plot saved to sensor_plot.png")

# Show statistics
print(f"\nSensor statistics:")
print(f"  Mean: {df[1].mean():.6f}")
print(f"  Std Dev: {df[1].std():.6f}")
print(f"  Min: {df[1].min():.6f} at row {df[1].idxmin()}")
print(f"  Max: {df[1].max():.6f} at row {df[1].idxmax()}")

# Look for last N rows to identify where LBO likely happens
print(f"\nLast 50 rows (where LBO likely occurs):")
print(df.tail(50).to_string())

# Find rows with biggest drops (negative spikes)
df['change'] = df[1].diff()
biggest_drops = df.nsmallest(10, 'change')
print(f"\nBiggest negative changes (spikes down):")
print(biggest_drops[['change']])
print(f"\nThese row indices likely indicate LBO event:")
print(f"  {biggest_drops.index.tolist()}")
