# -*- coding: utf-8 -*-
"""
BÀI 2: Tìm giá trị cực tiểu của hàm số g(x) = (1/3)x^3 - x
Sử dụng thuật toán Gradient Descent (theo mẫu bài giảng)

Công thức tổng quát:
    x(t+1) = x(t) - eta * g'(x(t))

Đạo hàm: g'(x) = 3.(1/3).x^2 - 1.x^0 = x^2 - 1

Lưu ý: g(x) có 1 điểm cực đại tại x=-1 và 1 điểm cực tiểu tại x=1
(vì g''(x) = 2x, tại x=1 thì g''>0 -> cực tiểu, tại x=-1 thì g''<0 -> cực đại).
Vì vậy, tuỳ điểm khởi tạo x0, thuật toán có thể hội tụ về các điểm khác nhau,
thậm chí phân kỳ (chạy ra vô cực) nếu x0 nằm quá xa về phía bên trái (x0 < -1),
vì bản thân hàm g(x) không bị chặn dưới khi x -> -vô cực.
"""

import numpy as np


def grad(x):
    """Tính đạo hàm g'(x) = x^2 - 1"""
    return x**2 - 1


def cost(x):
    """Tính giá trị của hàm số g(x) = (1/3)x^3 - x"""
    return (1 / 3) * x**3 - x


def myGD1(x0, eta):
    """
    Thuật toán Gradient Descent
    Đầu vào:
        x0  : điểm xuất phát
        eta : learning rate (tốc độ học)
    Thuật toán dừng lại khi đạo hàm có độ lớn đủ nhỏ
    """
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta * grad(x[-1])
        if abs(x_new) > 1e6:  # phát hiện phân kỳ (chạy ra vô cực)
            print(f"  [Cảnh báo] Điểm khởi tạo x0={x0} làm thuật toán phân kỳ, dừng lại.")
            return (x, it)
        if abs(grad(x_new)) < 1e-3:  # just a small number
            break
        x.append(x_new)
    return (x, it)


if __name__ == "__main__":
    # Thử tìm nghiệm với các điểm khởi tạo khác nhau,
    # cùng learning rate eta = 0.1
    (x1, it1) = myGD1(-0.5, .1)  # nằm bên phải x=-1 -> hội tụ về cực tiểu x=1
    (x2, it2) = myGD1(5, .1)     # nằm bên phải x=1 -> hội tụ về cực tiểu x=1
    (x3, it3) = myGD1(-5, .1)    # nằm bên trái x=-1 -> PHÂN KỲ (minh hoạ)

    print('Solution x1 = %f, cost = %f, after %d iterations'
          % (x1[-1], cost(x1[-1]), it1))
    print('Solution x2 = %f, cost = %f, after %d iterations'
          % (x2[-1], cost(x2[-1]), it2))
    print('Solution x3 = %f, cost = %f, after %d iterations (phân kỳ - không phải nghiệm)'
          % (x3[-1], cost(x3[-1]), it3))
