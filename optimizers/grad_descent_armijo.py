import numpy as np

from optimizers.optimizer_base import Criteria
from optimizers.grad_descent_base import grad_descent_base


class grad_descent_armijo(grad_descent_base):
    def __init__(self, fun, grad_f, tol: float, starting_pos: np.ndarray, criteria: Criteria = Criteria.NORM_GRAD,
                 max_iter: int = 1000, do_history: bool = False, c: float = 0.1, q: float = 0.7,
                 alpha0: float = 1, max_backtrack: int = 100):
        super().__init__(fun, grad_f, tol, starting_pos, "Armijo", criteria, max_iter, do_history)
        self.c = c
        self.q = q
        self.max_backtrack = max_backtrack
        self.alpha0 = alpha0

    def calculate_alpha(self) -> float:
        norm_gk_sqr = np.dot(self.grad_cur_pos, self.grad_cur_pos)
        alpha_i = self.alpha0
        pk = -self.grad_cur_pos
        for i in range(self.max_backtrack):
            if self.f_tilde(pk, alpha_i) <= self.f_cur_pos - self.c * alpha_i * norm_gk_sqr:
                break
            alpha_i *= self.q
        return alpha_i
