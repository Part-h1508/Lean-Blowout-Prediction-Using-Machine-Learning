import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('file1')
print(df.head())
print(df.columns)
print(df.shape)
"""
# EDA
# Y is target variable
# joint plot only compares two variables
sns.pairplot(df, kind='scatter', plot_kws={'alpha': 0.4})
plt.show()


sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=df, scatter_kws={'alpha': 0.4})
plt.show()

"""

import statsmodels.api as sm

x = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = df['Yearly Amount Spent']

# Add constant for intercept
x = sm.add_constant(x)

# Split data (70% train, 30% test)
split_idx = int(len(x) * 0.7)
x_train, x_test = x[:split_idx], x[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# train the model
lm = sm.OLS(y_train, x_train).fit()

#predictions
predictions = lm.predict(x_test)
sns.scatterplot(x=predictions, y=y_test)
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Linear Regression: Predicted vs Actual')
plt.show()
plt.show()
