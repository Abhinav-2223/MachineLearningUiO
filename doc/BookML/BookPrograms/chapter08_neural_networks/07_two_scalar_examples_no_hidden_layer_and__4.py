"""Chapter 8: listing 7, from the section on two scalar examples no hidden layer and .

Extracted from doc/BookML/chapter8.tex.
"""

import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

def cost(params, x, y):
    w_1, b_1, w_2, b_2 = params
    a_1 = jax.nn.sigmoid(x @ w_1 + b_1)        # the forward pass, nothing else
    a_2 = a_1 @ w_2 + b_2
    return 0.5*jnp.sum((a_2 - y)**2)

grad = jax.jit(jax.grad(cost))                 # the backward pass, generated

np.random.seed(0)
x = jnp.array([[4.0]]); y = 2*x + 1.0
params = [jnp.array(np.random.randn(1, 1)), jnp.zeros(1) + 0.01,
          jnp.array(np.random.randn(1, 1)), jnp.zeros(1) + 0.01]

gamma = 0.1
for i in range(50):
    if i % 10 == 0 or i == 49:
        print(f"iteration {i:2d}: cost = {cost(params, x, y):.6f}")
    g = grad(params, x, y)
    params = [p - gamma*gp for p, gp in zip(params, g)]
w_1, b_1, w_2, b_2 = params
print("w_1 =", w_1.ravel(), " b_1 =", b_1)
print("w_2 =", w_2.ravel(), " b_2 =", b_2)
