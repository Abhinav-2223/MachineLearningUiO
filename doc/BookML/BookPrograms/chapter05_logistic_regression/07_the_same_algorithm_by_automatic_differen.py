"""Chapter 5: listing 7, from the section on the same algorithm by automatic differen.

Extracted from doc/BookML/chapter5.tex.
"""

import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

def cross_entropy_one(theta0, theta1, x, y):
    """The trace of Eq. (5.adtrace) for one observation, one line per node."""
    v1 = theta1 * x
    v2 = v1 + theta0
    v3 = jnp.exp(v2)
    v4 = 1.0 + v3
    v5 = jnp.log(v4)
    v6 = y * v2
    return v5 - v6

def reverse_sweep(theta0, theta1, x, y):
    """Eq. (5.adreverse) by hand: forward evaluation onto the tape, then back."""
    v1 = theta1 * x; v2 = v1 + theta0; v3 = np.exp(v2); v4 = 1.0 + v3   # tape
    b7 = 1.0
    b5, b6 = b7, -b7
    b4 = b5 / v4
    b3 = b4
    b2 = b3 * v3 + b6 * y                  # fan-out: two successors, one sum
    b1 = b2
    return float(b2), float(b1 * x)        # dC/dtheta0, dC/dtheta1

x, y = 2.0, 1.0
theta = jnp.array([0.5, 0.25])
cost_one = lambda th: cross_entropy_one(th[0], th[1], x, y)
print("C_i =", float(cost_one(theta)))
for k in range(2):                         # forward mode: one sweep per parameter
    _, tangent = jax.jvp(cost_one, (theta,), (jnp.eye(2)[k],))
    print(f"forward sweep, seed e_{k}: dC/dtheta_{k} = {float(tangent):.6f}")
print("reverse sweep, jax.grad:", np.asarray(jax.grad(cost_one)(theta)))
print("reverse sweep, by hand: ", reverse_sweep(0.5, 0.25, x, y))

# the whole data set: mean cross entropy, Eq. (5.crossentropycompact) / n
def cost(theta, X, y):
    z = X @ theta
    return jnp.mean(jnp.logaddexp(0.0, z) - y * z)

rng = np.random.default_rng(2024)
n = 5000
for p in (20, 200, 1000):
    X = jnp.asarray(np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))]))
    yb = jnp.asarray(rng.integers(0, 2, n).astype(float))
    th = jnp.zeros(p)
    value = jax.jit(cost)
    reverse = jax.jit(jax.grad(cost))          # one reverse sweep
    forward = jax.jit(jax.jacfwd(cost))        # p forward sweeps
    for fn in (value, reverse, forward):       # compile before timing
        fn(th, X, yb).block_until_ready()
    p_hat = jax.nn.sigmoid(X @ th)
    g_hand = X.T @ (p_hat - yb) / n            # Eq. (5.gradient) / n
    print(f"p = {p:4d}: max|reverse - hand| = "
          f"{float(jnp.max(jnp.abs(reverse(th, X, yb) - g_hand))):.1e}", end="")
    for name, fn in (("value", value), ("reverse", reverse), ("forward", forward)):
        t0 = time.perf_counter()
        for _ in range(10):
            fn(th, X, yb).block_until_ready()
        print(f"   {name} {1e3 * (time.perf_counter() - t0) / 10:7.2f} ms", end="")
    print()
