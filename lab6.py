import numpy as np
import matplotlib.pyplot as plt

# Create synthetic data
np.random.seed(0)
X = np.linspace(0, 10, 100)
y = np.sin(X) + np.random.normal(0, 0.1, 100)  # non-linear + noise
X = X.reshape(-1, 1)

# Add bias term (column of 1s) for linear regression
def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])

# Gaussian kernel function
def kernel(x, x_i, tau):
    return np.exp(-np.sum((x - x_i)**2) / (2 * tau**2))

# LWR prediction for a single query point
def predict_point(x_query, X, y, tau):
    m = X.shape[0]
    W = np.eye(m)
    for i in range(m):
        W[i, i] = kernel(x_query, X[i], tau)
    
    X_bias = add_bias(X)
    x_query_bias = np.array([1, x_query[0]])
    
    # Theta = (Xᵀ W X)⁻¹ Xᵀ W y
    try:
        theta = np.linalg.pinv(X_bias.T @ W @ X_bias) @ X_bias.T @ W @ y
    except np.linalg.LinAlgError:
        return 0
    
    return x_query_bias @ theta

# Predict for all points
def locally_weighted_regression(X, y, tau):
    y_pred = []
    for x in X:
        y_pred.append(predict_point(x, X, y, tau))
    return np.array(y_pred)

# Set bandwidth (tau): lower = more local
tau = 0.5
y_pred = locally_weighted_regression(X, y, tau)

# Plot original data + LWR curve
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='lightgray', label='Data', alpha=0.6)
plt.plot(X, y_pred, color='red', linewidth=2, label=f'LWR (tau={tau})')
plt.title("Locally Weighted Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
