# -*- coding: utf-8 -*-
"""
BÀI 1: Tìm giá trị cực tiểu của hàm số f(x) = x^2 - 2
Sử dụng thuật toán Gradient Descent (theo mẫu bài giảng)

Công thức tổng quát:
    x(t+1) = x(t) - eta * f'(x(t))

Đạo hàm: f'(x) = 2x
"""

import numpy as np


def grad(x):
    """Tính đạo hàm f'(x) = 2x"""
    return 2 * x


def cost(x):
    """Tính giá trị của hàm số f(x) = x^2 - 2"""
    return x**2 - 2


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
        if abs(grad(x_new)) < 1e-3:  # just a small number
            break
        x.append(x_new)
    return (x, it)


if __name__ == "__main__":
    # Thử tìm nghiệm với các điểm khởi tạo khác nhau,
    # cùng learning rate eta = 0.1
    (x1, it1) = myGD1(-5, .1)
    (x2, it2) = myGD1(5, .1)

    print('Solution x1 = %f, cost = %f, after %d iterations'
          % (x1[-1], cost(x1[-1]), it1))
    print('Solution x2 = %f, cost = %f, after %d iterations'
          % (x2[-1], cost(x2[-1]), it2))
