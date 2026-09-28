import numpy as np

A = np.array([
    [1, 2, 3, 5],
    [-2, 4, 3, -4],
    [3, 2, -1, 6],
    [4, -3, 2, 1]
], dtype=float)

B = np.array([7, 12, -3, -3], dtype=float)


def matrix_method(A, B):
    # Перевірка, чи матриця вироджена
    det = np.linalg.det(A)
    if np.isclose(det, 0):
        raise ValueError("Визначник матриці дорівнює нулю, оберненої матриці не існує.")

    # Знаходження оберненої матриці
    A_inv = np.linalg.inv(A)

    # Множення оберненої матриці на вектор вільних членів
    X = np.dot(A_inv, B)
    return X, A_inv


x_matrix, A_inverse = matrix_method(A, B)

print("Розв'язок матричним методом:")
for i, x in enumerate(x_matrix):
    print(f"x_{i + 1} = {x:.4f}")
