import numpy as np

C_raw = np.array([
    [0.07, -0.08, 0.11, -0.18],
    [0.14, -0.42, 0.00, 0.21],
    [0.13, 0.31, 0.00, -0.19],
    [0.08, -0.33, 0.00, 0.28]
], dtype=float)

d_raw = np.array([-0.51, 1.18, -1.02, -0.28], dtype=float)

# Обов'язкове перетворення для нулів на головній діагоналі матриці C
C = np.zeros_like(C_raw)
d = np.zeros_like(d_raw)

for i in range(len(d_raw)):
    factor = 1.0 - C_raw[i, i]
    for j in range(len(d_raw)):
        if i != j:
            C[i, j] = C_raw[i, j] / factor
    d[i] = d_raw[i] / factor


def seidel_method(C, d, epsilon=1e-4, max_iter=1000):
    n = len(d)
    x = np.copy(d)  # Початкове наближення
    q = np.linalg.norm(C, ord=np.inf)

    for k in range(1, max_iter + 1):
        x_new = np.copy(x)
        for i in range(n):
            # Використовуємо нові значення x_new для j < i та старі значення x для j > i
            s1 = sum(C[i, j] * x_new[j] for j in range(i))
            s2 = sum(C[i, j] * x[j] for j in range(i + 1, n))
            x_new[i] = s1 + s2 + d[i]

        diff_norm = np.max(np.abs(x_new - x))

        # Визначення критерію зупинки
        tolerance = epsilon if q <= 0.5 else (((1 - q) / q) * epsilon if q < 1 else epsilon)

        if diff_norm <= tolerance:
            return x_new, k

        x = x_new

    raise ValueError("Метод Зейделя не зійшовся за задану кількість ітерацій.")


x_seidel, iters = seidel_method(C, d, epsilon=1e-4)

print(f"Розв'язок методом Зейделя (за {iters} ітерацій):")
for i, x_val in enumerate(x_seidel):
    print(f"x_{i + 1} = {x_val:.4f}")

# Перевірка знайденого розв'язку шляхом підстановки у вихідну СЛАР x = C_raw * x + d_raw
print("\n--- Перевірка підстановкою ---")
verification = np.dot(C_raw, x_seidel) + d_raw
for i in range(len(verification)):
    print(
        f"Рівняння {i + 1}: Знайдене x_{i + 1} = {x_seidel[i]:.4f} | Обчислене значення = {verification[i]:.4f} | Різниця = {abs(x_seidel[i] - verification[i]):.4e}")
