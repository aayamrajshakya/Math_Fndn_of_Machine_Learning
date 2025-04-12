import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

X, y = make_moons(noise=0.2, n_samples=400, random_state=12)

plt.scatter(X[y==0, 0], X[y==0, 1], color='red', marker='^', alpha=0.5)

plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='o', alpha=0.5)

plt.tight_layout()
plt.show()

