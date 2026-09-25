"""Chapter 8: listing 4, from the section on two scalar examples no hidden layer and .

Extracted from doc/BookML/chapter8.tex.
"""

import numpy as np

def sigma(z):  return 1.0/(1.0+np.exp(-z))
def dsigma(z): s = sigma(z); return s*(1.0-s)

# no hidden layer: x -> z1 = w1 x + b1 -> a1 = sigma(z1) -> C = (a1-y)^2/2
def grad_one(w1, b1, x, y):
    z1 = w1*x + b1; a1 = sigma(z1)                 # forward pass, store z1, a1
    # error delta1 = (a1-y) sigma'(z1)
    delta1 = (a1 - y)*dsigma(z1)
    return 0.5*(a1-y)**2, delta1*x, delta1, delta1 # C, dC/dw1, dC/db1, delta1

# one hidden layer: x -> (w1,b1) -> a1 -> (w2,b2) -> a2 -> C = (a2-y)^2/2
def grad_two(w1, b1, w2, b2, x, y):
    z1 = w1*x + b1; a1 = sigma(z1)                 # forward pass
    z2 = w2*a1 + b2; a2 = sigma(z2)
    delta2 = (a2 - y)*dsigma(z2)                   # output error
    # delta1 = delta2 w2 sigma'(z1)
    delta1 = delta2*w2*dsigma(z1)
    return (0.5*(a2-y)**2, delta1*x, delta1, delta2*a1, delta2, delta1, delta2)

x, y = 2.0, 1.0
C, gw1, gb1, d1 = grad_one(0.25, 0.5, x, y)
print(f"no hidden layer : C = {C:.4f}  delta1 = {d1:.4f}")
print(f"   dC/dw1 = {gw1:.4f}  dC/db1 = {gb1:.4f}")
C, gw1, gb1, gw2, gb2, d1, d2 = grad_two(0.25, 0.5, 1.0, -0.5, x, y)
print(f"one hidden layer: C = {C:.4f}  delta2 = {d2:.4f}  delta1 = {d1:.4f}")
print(f"   dC/dw1 = {gw1:.4f}  dC/db1 = {gb1:.4f}", end="")
print(f"  dC/dw2 = {gw2:.4f}  dC/db2 = {gb2:.4f}")
