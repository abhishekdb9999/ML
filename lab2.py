import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset (default = Iris)
path = input("Enter CSV path (or press Enter to use Iris dataset): ")
if path.strip() == "":
    path = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"

df = pd.read_csv(path)

# Pick 2 numerical columns automatically
num_cols = df.select_dtypes(include='number').columns[:2]
col1, col2 = num_cols[0], num_cols[1]

# --- Scatter Plot ---
plt.figure(figsize=(6, 5))
sns.scatterplot(x=df[col1], y=df[col2])
plt.title(f'Scatter Plot: {col1} vs {col2}')
plt.xlabel(col1)
plt.ylabel(col2)
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Pearson Correlation ---
pearson_corr = df[[col1, col2]].corr(method='pearson').iloc[0, 1]
print(f"Pearson Correlation ({col1} vs {col2}): {pearson_corr:.4f}")

# --- Covariance Matrix ---
print("\nCovariance Matrix:")
print(df[num_cols].cov())

# --- Correlation Matrix ---
print("\nCorrelation Matrix:")
print(df[num_cols].corr())

# --- Heatmap ---
plt.figure(figsize=(6, 5))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.tight_layout()
plt.show()
