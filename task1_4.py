import numpy as np

A = np.array([
    [ 1,  2,  3,  5],
    [-2,  4,  3, -4],
    [ 3,  2, -1,  6],
    [ 4, -3,  2,  1]
], dtype=float)

B = np.array([7, 12, -3, -3], dtype=float)

def lu_decomposition_method(A, B):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    # Крок 1: Побудова матриць L та U (алгоритм Дулітла)
    for i in range(n):
        L[i, i] = 1.0
        for k in range(i, n):
            U[i, k] = A[i, k] - sum(L[i, j] * U[j, k] for j in range(i))
        for k in range(i + 1, n):
            L[k, i] = (A[k, i] - sum(L[k, j] * U[j, i] for j in range(i))) / U[i, i]

    # Крок 2: Розв'язання L * Y = B (прямий хід)
    Y = np.zeros(n)
    for i in range(n):
        Y[i] = B[i] - sum(L[i, j] * Y[j] for j in range(i))

    # Крок 3: Розв'язання U * X = Y (зворотний хід)
    X = np.zeros(n)
    for i in range(n - 1, -1, -1):
        X[i] = (Y[i] - sum(U[i, j] * X[j] for j in range(i + 1, n))) / U[i, i]

    return X

x_lu = lu_decomposition_method(A, B)

print("Розв'язок методом LU-розкладу:")
for i, x in enumerate(x_lu):
    print(f"x_{i+1} = {x:.4f}")
