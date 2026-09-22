"""Chapter 4: listing 6, from the section on stochastic gradient descent.

Extracted from doc/BookML/chapter4.tex.
"""

import numpy as np

n, p, rho, sigma = 100_000, 20, 0.9, 1.0
rng = np.random.default_rng(2026)
z = rng.normal(size=(n, p))
X = np.empty_like(z)
X[:, 0] = z[:, 0]
for j in range(1, p):                     # AR(1)-correlated columns: kappa = 213 after standardising
    X[:, j] = rho * X[:, j - 1] + np.sqrt(1 - rho**2) * z[:, j]
X = (X - X.mean(axis=0)) / X.std(axis=0)
theta_true = rng.normal(size=p)
y = X @ theta_true + sigma * rng.normal(size=n)

theta_hat = np.linalg.solve(X.T @ X, X.T @ y)          # np^2 + p^3/3 flops
c_hat = np.mean((X @ theta_hat - y)**2)
excess = lambda th: np.mean((X @ th - y)**2) - c_hat
eigs = np.linalg.eigvalsh((2.0 / n) * X.T @ X)
gamma_star = 2.0 / (eigs.max() + eigs.min())
flops_per_point = 4 * p                                # Eq. (4.flops)

theta = np.zeros(p)                                    # gradient descent: 4np flops per step
for k in range(1, 1001):
    theta -= gamma_star * (2.0 / n) * X.T @ (X @ theta - y)
    if excess(theta) < sigma**2 * p / n:
        print(f"gradient descent reaches sigma^2 p/n after {k} steps = {k * n * flops_per_point:.1e} flops")
        break

M, t, theta = 32, 0, np.zeros(p)                       # SGD: 4Mp flops per update, 4np per epoch
for epoch in range(1, 6):
    for b in np.array_split(rng.permutation(n), n // M):
        t += 1
        gamma = 20.0 / (t + 1000.0)                        # Eq. (4.timedecay)
        theta -= gamma * (2.0 / len(b)) * X[b].T @ (X[b] @ theta - y[b])
    print(f"SGD epoch {epoch}: excess cost {excess(theta):.2e} after {t * M * flops_per_point:.1e} flops")
