import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV (default fallback)
path = input("Enter CSV path (or press Enter to use sample): ")
if path.strip() == "":
    path = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"

df = pd.read_csv(path)

# Auto-select first numeric and categorical column
num_col = df.select_dtypes(include='number').columns[0]
cat_col = df.select_dtypes(include='object').columns[0]

data = df[num_col].dropna()

# --- Stats ---
print(f"\nStats for '{num_col}':")
print(f"Mean: {data.mean():.2f}")
print(f"Median: {data.median():.2f}")
print(f"Mode: {data.mode().iloc[0]:.2f}")
print(f"Std Dev: {data.std():.2f}")
print(f"Variance: {data.var():.2f}")
print(f"Range: {data.max() - data.min():.2f}")

# --- IQR Outliers ---
Q1, Q3 = data.quantile([0.25, 0.75])
IQR = Q3 - Q1
outliers = data[(data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)]
print(f"\nOutliers in '{num_col}':")
print(outliers.values)

# --- Plots ---
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.histplot(data, kde=True)
plt.title(f'Histogram of {num_col}')

plt.subplot(1, 2, 2)
sns.boxplot(x=data)
plt.title(f'Boxplot of {num_col}')
plt.tight_layout()
plt.show()

# --- Category Frequency ---
cat_freq = df[cat_col].value_counts()

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
cat_freq.plot(kind='bar', color='skyblue')
plt.title(f'Bar Chart of {cat_col}')

plt.subplot(1, 2, 2)
cat_freq.plot(kind='pie', autopct='%1.1f%%')
plt.title(f'Pie Chart of {cat_col}')
plt.ylabel('')
plt.tight_layout()
plt.show()
