"""Chapter 8: listing 5, from the section on two scalar examples no hidden layer and .

Extracted from doc/BookML/chapter8.tex.
"""

import numpy as np

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def forwardpropagation(x):
    z_1 = np.matmul(x, w_1) + b_1      # pre-activation of the hidden layer
    a_1 = sigmoid(z_1)                 # hidden activation
    z_2 = np.matmul(a_1, w_2) + b_2    # pre-activation of the output layer
    a_2 = z_2                          # linear output
    return a_1, a_2

def backpropagation(x, y):
    a_1, a_2 = forwardpropagation(x)
    delta_2 = a_2 - y                  # linear output: f' = 1
    # delta_2 w_2 sigma'(z_1)
    delta_1 = np.matmul(delta_2, w_2.T) * a_1 * (1 - a_1)
    dW2 = np.matmul(a_1.T, delta_2);  db2 = np.sum(delta_2, axis=0)
    dW1 = np.matmul(x.T, delta_1);    db1 = np.sum(delta_1, axis=0)
    return dW2, db2, dW1, db1, 0.5*np.sum((a_2 - y)**2)

np.random.seed(0)
x = np.array([[4.0]])                  # one sample, one feature: shape (1, 1)
y = 2*x + 1.0                          # target 9
n_features, n_hidden, n_outputs = 1, 1, 1
w_1 = np.random.randn(n_features, n_hidden); b_1 = np.zeros(n_hidden) + 0.01
w_2 = np.random.randn(n_hidden, n_outputs); b_2 = np.zeros(n_outputs) + 0.01

gamma = 0.1
for i in range(50):
    dW2, db2, dW1, db1, cost = backpropagation(x, y)
    if i % 10 == 0 or i == 49:
        print(f"iteration {i:2d}: cost = {cost:.6f}")
    w_2 -= gamma*dW2;  b_2 -= gamma*db2
    w_1 -= gamma*dW1;  b_1 -= gamma*db1
print("w_1 =", w_1.ravel(), " b_1 =", b_1)
print("w_2 =", w_2.ravel(), " b_2 =", b_2)
