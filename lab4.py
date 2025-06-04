import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Standardize features (important for distance-based models)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Function to evaluate k-NN
def evaluate_knn(k, weights='uniform'):
    model = KNeighborsClassifier(n_neighbors=k, weights=weights)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"\nK={k}, Weights='{weights}':")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Score: {f1:.4f}")

# Regular k-NN
print("🔍 Regular k-NN:")
for k in [1, 3, 5]:
    evaluate_knn(k, weights='uniform')

# Weighted k-NN (weight = 1 / d^2)
print("\n⚖️ Weighted k-NN (1/d²):")
for k in [1, 3, 5]:
    evaluate_knn(k, weights='distance')
