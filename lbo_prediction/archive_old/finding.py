import pandas as pd
import matplotlib.pyplot as plt
import os

# Change to script directory so relative paths work
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

df = pd.read_excel("data/90.xlsx", header=None)

signal = df[1]

rolling_mean = signal.rolling(500).mean()
rolling_std = signal.rolling(500).std()

plt.figure(figsize=(12,6))

plt.plot(rolling_mean, label='Rolling Mean')
plt.plot(rolling_std, label='Rolling Std')

plt.legend()
plt.grid(True)
plt.show()