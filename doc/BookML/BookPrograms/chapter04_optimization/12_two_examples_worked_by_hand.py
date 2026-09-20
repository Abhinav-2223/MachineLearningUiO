"""Chapter 4: listing 12, from the section on two examples worked by hand.

Extracted from doc/BookML/chapter4.tex.
"""

import math
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

def sqrt(d):                            # one more elementary function for Dual
    r = math.sqrt(d.val)
    return Dual(r, d.dot / (2.0 * r))

def f_chain(x):                         # Eq. (4.adchainfun)
    return exp(x**2)

def f_fanout(x):                        # Eq. (4.adfanoutfun)
    a = x**2
    return sqrt(a + exp(a))

x = 1.0
# forward mode: seed the tangent of x with 1 and read the dual part
print("chain,   forward:", f_chain(Dual(x, 1.0)).dot, " exact:", 2 * x * math.exp(x**2))
print("fan-out, forward:", f_fanout(Dual(x, 1.0)).dot)

# reverse mode: jax.grad is one reverse sweep, Eq. (4.adfanoutreverse)
def f_fanout_jax(x):
    a = x**2
    return jnp.sqrt(a + jnp.exp(a))

print("fan-out, reverse:", float(jax.grad(f_fanout_jax)(x)))
print("fan-out, closed: ", x * (1 + math.exp(x**2)) / math.sqrt(x**2 + math.exp(x**2)))
