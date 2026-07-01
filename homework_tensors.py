import torch

# 1.1 Создание тензоров (7 баллов)
print("=== 1.1 Создание тензоров ===")
tensor_rand = torch.rand(3, 4)
tensor_zeros = torch.zeros(2, 3, 4)
tensor_ones = torch.ones(5, 5)
tensor_arange = torch.arange(16).reshape(4, 4)

print(f"Случайный 3x4: {tensor_rand.shape}")
print(f"Нулевой 2x3x4: {tensor_zeros.shape}")
print(f"Единичный 5x5: {tensor_ones.shape}")
print(f"0-15 в 4x4: {tensor_arange}")
print()

# 1.2 Операции с тензорами (6 баллов)
print("=== 1.2 Операции с тензорами ===")
A = torch.rand(3, 4)
B = torch.rand(4, 3)

A_T = A.T  # Транспонирование
AB = A @ B  # Матричное умножение
elementwise = A * B.T  # Поэлементное умножение
sum_A = A.sum()  # Сумма всех элементов

print(f"A: {A.shape}")
print(f"B: {B.shape}")
print(f"Транспонирование A: {A_T.shape}")
print(f"Матричное умножение A@B: {AB.shape}")
print(f"Поэлементное умножение A*B.T: {elementwise.shape}")
print(f"Сумма A: {sum_A.item():.4f}")
print()

# 1.3 Индексация и срезы (6 баллов)
print("=== 1.3 Индексация и срезы ===")
tensor_5x5x5 = torch.arange(125).reshape(5, 5, 5)

first_row = tensor_5x5x5[0, :, :]  # Первая строка (все столбцы и глубины)
last_column = tensor_5x5x5[:, :, -1]  # Последний столбец
center_2x2 = tensor_5x5x5[2:4, 2:4, 2:4]  # Подматрица 2x2 из центра
even_indices = tensor_5x5x5[::2, ::2, ::2]  # Все элементы с четными индексами

print(f"Первая строка: {first_row.shape}")
print(f"Последний столбец: {last_column.shape}")
print(f"Центральная 2x2: {center_2x2.shape}")
print(f"Четные индексы: {even_indices.shape}")
print()

# 1.4 Работа с формами (6 баллов)
print("=== 1.4 Работа с формами ===")
tensor_24 = torch.arange(24)

shapes = [
    (2, 12),
    (3, 8),
    (4, 6),
    (2, 3, 4),
    (2, 2, 2, 3)
]

for shape in shapes:
    reshaped = tensor_24.reshape(shape)
    print(f"Reshape to {shape}: {reshaped.shape} (элементов: {reshaped.numel()})")