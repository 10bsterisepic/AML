import numpy as np
import matplotlib.pyplot as plt
# Data & priors
np.random.seed(42)
mu_real, sigma_real, N = 5, 2, 100
data = np.random.normal(mu_real, sigma_real, N)
mu, sigma = 0, 1  # initial guesses
# Gibbs Sampling
iters = 2000
mu_samples, sigma_samples = [], []
for _ in range(iters):
  mu = np.random.normal(np.mean(data), sigma / np.sqrt(N))
  sigma = np.sqrt(1 / np.random.gamma(N / 2, 2 / np.sum((data - mu) ** 2)))
  mu_samples.append(mu)
  sigma_samples.append(sigma)
# Trace plots
plt.subplot(2, 1, 1)
plt.plot(mu_samples)
plt.axhline(mu_real, color='r')
plt.title("Mu Trace")
plt.subplot(2, 1, 2)
plt.plot(sigma_samples)
plt.axhline(sigma_real, color='r')
plt.title("Sigma Trace")
plt.tight_layout()
plt.show()
print(f"Estimated mu: {np.mean(mu_samples):.2f}, sigma: {np.mean(sigma_samples):.2f}")
