"""Chapter 8: listing 6, from the section on two scalar examples no hidden layer and .

Extracted from doc/BookML/chapter8.tex.
"""

import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

def C_one(p, x, y):                    # forward pass of the single neuron ...
    return 0.5*(jax.nn.sigmoid(p[0]*x + p[1]) - y)**2
def C_two(p, x, y):                    # ... and with the hidden neuron
    a1 = jax.nn.sigmoid(p[0]*x + p[1])
    return 0.5*(jax.nn.sigmoid(p[2]*a1 + p[3]) - y)**2

x, y = 2.0, 1.0
p1, p2 = jnp.array([0.25, 0.5]), jnp.array([0.25, 0.5, 1.0, -0.5])
print("no hidden layer,  jax.grad:", jax.grad(C_one)(p1, x, y))
print("one hidden layer, jax.grad:", jax.grad(C_two)(p2, x, y))
# forward mode, one sweep per parameter, for comparison
seeds = jnp.eye(2)
print("no hidden layer,  jax.jvp: ",
      [float(jax.jvp(C_one, (p1, x, y), (s, 0.0, 0.0))[1]) for s in seeds])
