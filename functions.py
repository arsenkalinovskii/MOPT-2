import numpy as np


class func():
    def __init__(self, f, grad_f, name):
        self.f = f
        self.grad_f = grad_f
        self.name = name

    def __str__(self):
        return self.name


def f_(x):
    return x[0] ** 2 + x[1] ** 2


def grad_f_(x):
    return np.array([2 * x[0], 2 * x[1]])


f = func(f_, grad_f_, 'Good function')


def g_(x):
    return x[0] ** 2 + 100 * x[1] ** 2


def grad_g_(x):
    return np.array([2 * x[0], 200 * x[1]])


g = func(g_, grad_g_, 'Bad function')


def rosenbrock_(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def grad_rosenbrock_(x):
    dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2)
    dy = 200 * (x[1] - x[0] ** 2)
    return np.array([dx, dy])


rosenbrock = func(rosenbrock_, grad_rosenbrock_, 'Rosenbrock function')


def himmelblau_(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def grad_himmelblau_(x):
    dx = 4 * x[0] * (x[0] ** 2 + x[1] - 11) + 2 * (x[0] + x[1] ** 2 - 7)
    dy = 2 * (x[0] ** 2 + x[1] - 11) + 4 * x[1] * (x[0] + x[1] ** 2 - 7)
    return np.array([dx, dy])


himmelblau = func(himmelblau_, grad_himmelblau_, 'Himmelblau function')


def ackley_(x):
    term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2)))
    term2 = -np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1])))
    return term1 + term2 + np.e + 20


def grad_ackley_(x):
    r_sq = x[0] ** 2 + x[1] ** 2
    # Защита от деления на ноль в точке глобального минимума (0,0)
    if r_sq < 1e-12:
        return np.array([0.0, 0.0])

    term1_base = np.exp(-0.2 * np.sqrt(0.5 * r_sq))
    term2_base = np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1])))

    dx = (2 * x[0] * term1_base) / np.sqrt(0.5 * r_sq) + np.pi * np.sin(2 * np.pi * x[0]) * term2_base
    dy = (2 * x[1] * term1_base) / np.sqrt(0.5 * r_sq) + np.pi * np.sin(2 * np.pi * x[1]) * term2_base

    return np.array([dx, dy])


ackley = func(ackley_, grad_ackley_, 'Ackley function')
