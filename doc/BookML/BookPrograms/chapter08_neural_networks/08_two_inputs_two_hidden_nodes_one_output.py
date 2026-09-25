"""Chapter 8: listing 8, from the section on two inputs two hidden nodes one output.

Extracted from doc/BookML/chapter8.tex.
"""

import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

def sigmoid(z): return 1.0/(1.0+np.exp(-z))

def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1; A1 = sigmoid(Z1)         # hidden layer, samples in rows
    Z2 = A1 @ W2 + b2; A2 = Z2                  # linear output node
    return A1, A2

def backward(X, y, W1, b1, W2, b2):
    n = X.shape[0]
    A1, A2 = forward(X, W1, b1, W2, b2)
    delta2 = (A2 - y)/n                         # output error, f' = 1
    # delta_j^1 = delta^2 w_j^2 sigma'(z_j^1)
    delta1 = (delta2 @ W2.T) * A1*(1 - A1)
    dW2 = A1.T @ delta2;  db2 = delta2.sum(axis=0)
    dW1 = X.T @ delta1;   db1 = delta1.sum(axis=0)
    return dW1, db1, dW2, db2, 0.5*np.sum((A2 - y)**2)/n

def cost(params, X, y):                         # the same model for jax.grad
    W1, b1, W2, b2 = params
    A1 = jax.nn.sigmoid(X @ W1 + b1)
    return 0.5*jnp.mean((A1 @ W2 + b2 - y)**2)

rng = np.random.default_rng(41)
n = 100
X = rng.uniform(0, 1, size=(n, 2))
y = (X[:, 0]**2 + 3*X[:, 0]*X[:, 1] + X[:, 1]**2 + 5).reshape(-1, 1)

# w_{ij}^1, i input, j hidden
W1 = rng.standard_normal((2, 2)); b1 = np.zeros(2) + 0.01
W2 = rng.standard_normal((2, 1)); b2 = np.zeros(1) + 0.01   # w_{j}^2

g_hand = backward(X, y, W1, b1, W2, b2)[:4]
params = [jnp.array(p) for p in (W1, b1, W2, b2)]
g_jax = jax.grad(cost)(params, X, y)
print("max |hand - jax.grad| over the nine derivatives:",
      max(float(jnp.max(jnp.abs(h - j))) for h, j in zip(g_hand, g_jax)))

gamma = 0.5
for i in range(5001):
    dW1, db1, dW2, db2, C = backward(X, y, W1, b1, W2, b2)
    if i % 1000 == 0: print(f"iteration {i:4d}: cost = {C:.5f}")
    W1 -= gamma*dW1; b1 -= gamma*db1; W2 -= gamma*dW2; b2 -= gamma*db2
print("W1 =\n", W1.round(3), "\nb1 =", b1.round(3))
print("W2 =", W2.ravel().round(3), " b2 =", b2.round(3))
