
import numpy as np
import matplotlib.pyplot as plt

# Training data
X = np.array([[1, 1], [-1, -1], [0, 0.5], [0.1, 0.5], [0.2, 0.2], [0.9, 0.5]])
y = np.array([1, -1, -1, -1, 1, 1])

# Initial weight vector
w = np.array([1, 1])

# Iterate until convergence
while True:
    misclassified = False
    for i in range(len(X)):
        if y[i] * np.dot(w, X[i]) <= 0:
            w = w + y[i] * X[i]
            misclassified = True
    if not misclassified:
        break

# Plot data points
plt.scatter(X[:, 0], X[:, 1], c=y)

# Plot decision boundary
x1 = np.linspace(-1, 1, 100)
x2 = (-w[0] * x1) / w[1]
plt.plot(x1, x2)

plt.show()
