"""Chapter 4: listing 4, from the section on momentum.

Extracted from doc/BookML/chapter4.tex.
"""

import numpy as np

def quartic(x):
    """A low-order polynomial with two minima: f(x) = x^4 - 3x^2 + x."""
    return x**4 - 3.0 * x**2 + x

def quartic_grad(x):
    return 4.0 * x**3 - 6.0 * x + 1.0

def descend(grad, x0, gamma, beta=0.0, num_iters=40):
    """Gradient descent with momentum, Eq. (4.momentum); beta = 0 is plain gradient descent.
    Works for a scalar x as well as for a vector theta.  Returns every iterate."""
    x = np.array(x0, dtype=float)
    v = np.zeros_like(x)
    history = [x.copy()]
    for _ in range(num_iters):
        v = beta * v + gamma * grad(x)
        x = x - v
        history.append(x.copy())
    return np.array(history)

for beta in (0.0, 0.7):
    path = descend(quartic_grad, 2.0, gamma=0.05, beta=beta)
    print(f"beta = {beta}: x = {np.round(path[:6], 3)} ... ends at {path[-1]:.4f}, f = {quartic(path[-1]):.4f}")
