from abc import abstractmethod
from typing import List

from optimizers.optimizer_base import base_optimizer, Criteria
import numpy as np

class grad_descent_base(base_optimizer):
    def __init__(self, fun, grad_f, tol: float, starting_pos, name : str, criteria : Criteria = Criteria.NORM_GRAD, max_iter: int = 1000, do_history: bool = False):
        super().__init__(fun, grad_f, tol, starting_pos, name, criteria)
        self.history = []
        self.max_iter = max_iter
        self.do_history = do_history

        self.iteration_count = 0
        self.function_diff = None
        self.grad_cur_pos = None
        self.f_cur_pos = None
        self.point_diff = None

    def reached_tolerance(self) -> bool:
        if self.iteration_count >= self.max_iter:
            print("WARNING: reached iteration limit")
            return True
        if self.criteria == Criteria.NORM_GRAD:
            if self.grad_cur_pos is None or np.linalg.norm(self.grad_cur_pos) >= self.eps:
                return False
        elif self.criteria == Criteria.FUNCTION_DIFF:
            if self.function_diff is None or abs(self.function_diff) >= self.eps:
                return False
        elif self.criteria == Criteria.POINT_DIFF:
            if self.point_diff is None or np.linalg.norm(self.point_diff) >= self.eps:
                return False
        return True

    def find_minimum(self) -> tuple[np.ndarray, float]:
        while not self.reached_tolerance():
            self.iteration()
        return self.cur_pos, self.f_cur_pos

    def iteration(self):
        if not (self.point_diff is None):
            self.cur_pos += self.point_diff
        self.grad_cur_pos = self.calculate_grad(self.cur_pos)
        self.f_cur_pos = self.calculate_function(self.cur_pos)
        if self.do_history:
            self.history.append((self.cur_pos.copy(), self.f_cur_pos, self.grad_cur_pos))
        alpha = self.calculate_alpha()
        self.point_diff = -(self.grad_cur_pos * alpha)
        self.iteration_count += 1

    def reset(self):
        super().reset()
        self.iteration_count = 0
        self.function_diff = None
        self.grad_cur_pos = None
        self.f_cur_pos = None
        self.point_diff = None

    def f_tilde(self, pk, alpha : float) -> float:
        return self.calculate_function(self.step(alpha, pk))

    def step(self, alpha : float, direction):
        return self.cur_pos + alpha * direction

    @abstractmethod
    def calculate_alpha(self) -> float:
        pass