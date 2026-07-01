import torch

# 2.1 Простые вычисления с градиентами (8 баллов)
print("=== 2.1 Простые вычисления с градиентами ===")

# Создайте тензоры x, y, z с requires_grad=True
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = torch.tensor(1.0, requires_grad=True)

# Вычислите функцию: f(x,y,z) = x^2 + y^2 + z^2 + 2*x*y*z
f = x**2 + y**2 + z**2 + 2*x*y*z
print(f"f = {f.item():.2f}")

# Найдите градиенты по всем переменным
f.backward()
print(f"grad_x = {x.grad.item():.2f}")
print(f"grad_y = {y.grad.item():.2f}")
print(f"grad_z = {z.grad.item():.2f}")

# Проверка аналитически
print("\nПроверка аналитически:")
print(f"grad_x = {2*x.item() + 2*y.item()*z.item():.2f}")
print(f"grad_y = {2*y.item() + 2*x.item()*z.item():.2f}")
print(f"grad_z = {2*z.item() + 2*x.item()*y.item():.2f}")
print()

# 2.2 Градиент функции потерь (9 баллов)
print("=== 2.2 Градиент функции потерь ===")

def mse_gradients(x, y_true, w, b):
    w = torch.tensor(w, requires_grad=True)
    b = torch.tensor(b, requires_grad=True)
    loss = torch.mean((w * torch.tensor(x) + b - torch.tensor(y_true)) ** 2)
    loss.backward()
    return w.grad.item(), b.grad.item()

x = [1, 2, 3, 4]
y_true = [2, 4, 6, 8]
w = 1.0
b = 0.0

gw, gb = mse_gradients(x, y_true, w, b)
print(f"grad_w = {gw:.4f}, grad_b = {gb:.4f}")
print()

# 2.3 Цепное правило (8 баллов)
print("=== 2.3 Цепное правило ===")

def f(x):
    return torch.sin(x**2 + 1)

x = torch.tensor(2.0, requires_grad=True)
y = f(x)

print(f"f(2) = {y.item():.4f}")

grad = torch.autograd.grad(y, x)[0]
print(f"df/dx at x=2: {grad.item():.4f}")