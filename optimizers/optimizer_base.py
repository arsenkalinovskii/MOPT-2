from abc import abstractmethod, ABC
from cmath import nan
from enum import Enum

import numpy


class Criteria(Enum):
    NORM_GRAD = 0
    FUNCTION_DIFF = 1
    POINT_DIFF = 2
    SIGNIFICANT_DECREASE = 3


class base_optimizer(ABC):
    def __init__(self, fun, grad_f, tol: float, starting_pos: numpy.ndarray, name: str, criteria: Criteria):
        self.function = fun
        self.grad_f = grad_f
        self.eps = tol
        self.criteria = criteria
        self.cur_pos = starting_pos
        self.starting_pos = starting_pos
        self.name = name

        self.function_call_count = 0
        self.grad_call_count = 0

    def calculate_function(self, x) -> float:
        result = self.function(x)
        self.function_call_count += 1
        return result

    def calculate_grad(self, x) -> numpy.ndarray:
        result = self.grad_f(x)
        self.grad_call_count += 1
        return result

    def reset(self):
        self.cur_pos = self.starting_pos
        self.function_call_count = 0
        self.grad_call_count = 0

    def __str__(self):
        return self.name

    @abstractmethod
    def find_minimum(self) -> tuple[numpy.ndarray, float]:
        pass
