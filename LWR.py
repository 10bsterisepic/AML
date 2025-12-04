import numpy as np
import matplotlib.pyplot as plt

def gaussian_kernel(x, x0, tau):
    """Compute Gaussian kernel weights for point x0."""
    return np.exp(-np.sum((x - x0)*2, axis=1) / (2 * tau*2))

def locally_weighted_regression(X, y, x0, tau):
    """Perform LWR for a single query point x0."""
    m = len(X)
    W = np.eye(m)  # Weight matrix
    weights = gaussian_kernel(X, x0, tau)
    np.fill_diagonal(W, weights)

    # Normal equation: theta = (X^T W X)^(-1) X^T W y
    XTWX = X.T @ W @ X
    if np.linalg.det(XTWX) == 0.0:
        theta = np.linalg.pinv(XTWX) @ X.T @ W @ y
    else:
        theta = np.linalg.inv(XTWX) @ X.T @ W @ y

    return x0 @ theta

np.random.seed(42)
X = np.linspace(-3, 3, 50)
y = np.sin(X) + 0.3 * np.random.randn(50)

# Add bias term for linear regression form
X_mat = np.c_[np.ones(X.shape[0]), X]

X_test = np.linspace(-3, 3, 100)
X_test_mat = np.c_[np.ones(X_test.shape[0]), X_test]

tau = 0.5  # Bandwidth parameter
y_pred = np.array([locally_weighted_regression(X_mat, y, x0, tau) for x0 in X_test_mat])

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='red', label='Training data')
plt.plot(X_test, y_pred, color='blue', label=f'LWR Prediction (tau={tau})')
plt.title('Locally Weighted Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
