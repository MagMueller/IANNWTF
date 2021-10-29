import numpy as np

np.set_printoptions(suppress=True)
X = np.random.random((5, 5)).round(4)
print("Original Array:")
print(X)

print("Modified Array:")
x_mod = np.array([[round(X[i, j]**2, 4) if X[i, j] >
                 0.09 else 42 for j in range(X.shape[1])] for i in range(X.shape[0])])
print(x_mod)

print("4. column: \n {}".format(x_mod[:, 3]))
