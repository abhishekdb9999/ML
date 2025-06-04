import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Load Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y_true = data.target  # true labels (for comparison, not used in clustering)
feature_names = data.feature_names

# 2. Standardize features (important for k-means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply KMeans clustering
k = 2  # we know there are two classes (malignant/benign)
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# 4. Reduce to 2D with PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 5. Plot clusters vs actual classes
plt.figure(figsize=(12, 5))

# Cluster plot
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.7)
plt.title("K-Means Clusters (k=2)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

# True labels plot
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='viridis', s=50, alpha=0.7)
plt.title("True Labels (Malignant=0, Benign=1)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.tight_layout()
plt.show()
