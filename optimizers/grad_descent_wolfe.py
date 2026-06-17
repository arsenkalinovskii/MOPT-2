import numpy as np

from optimizers.optimizer_base import Criteria
from optimizers.grad_descent_base import grad_descent_base


class grad_descent_wolfe(grad_descent_base):
    def __init__(self, fun, grad_f, tol: float, starting_pos: np.ndarray, criteria: Criteria = Criteria.NORM_GRAD,
                 max_iter: int = 1000, do_history: bool = False, c1: float = 0.1, c2: float = 0.2,
                 q: float = 0.7, alpha0: float = 1, max_backtrack: int = 100):
        super().__init__(fun, grad_f, tol, starting_pos, "Wolfe", criteria, max_iter, do_history)
        self.c1 = c1
        self.c2 = c2
        self.max_backtrack = max_backtrack
        self.q = q
        self.alpha0 = alpha0

    def calculate_alpha(self) -> float:
        norm_gk_sqr = np.dot(self.grad_cur_pos, self.grad_cur_pos)
        alpha_i = self.alpha0
        alpha_prev = 0
        f_tilde_prev = self.f_cur_pos
        pk = -self.grad_cur_pos
        for i in range(self.max_backtrack):
            f_tilde = self.f_tilde(pk, alpha_i)
            if f_tilde > self.f_cur_pos - self.c1 * alpha_i * norm_gk_sqr \
                    or f_tilde >= f_tilde_prev:
                return self.zoom(alpha_prev, alpha_i, norm_gk_sqr)
            newpoint = self.step(alpha_i, pk)
            curvature = np.dot(self.calculate_grad(newpoint), pk)
            if abs(curvature) <= abs(self.c2 * np.dot(self.grad_cur_pos, pk)):
                return alpha_i
            if curvature >= 0:
                return self.zoom(alpha_i, self.alpha0, norm_gk_sqr)
            alpha_i, alpha_prev = (alpha_i + self.alpha0) / 2, alpha_i
        return alpha_i

    def zoom(self, alpha_lo: float, alpha_hi: float, norm_gk_sqr: float) -> float:
        pk = -self.grad_cur_pos
        f_tilde_lo = self.f_cur_pos
        for i in range(self.max_backtrack):
            alpha_j = (alpha_hi + alpha_lo) / 2
            f_tilde_j = self.f_tilde(pk, alpha_j)
            if f_tilde_j > self.f_cur_pos - self.c1 * alpha_j * norm_gk_sqr \
                    or f_tilde_j >= f_tilde_lo:
                alpha_hi = alpha_j
            else:
                newpoint = self.step(alpha_j, pk)
                curvature = np.dot(self.calculate_grad(newpoint), pk)
                if abs(curvature) <= abs(self.c2 * np.dot(self.grad_cur_pos, pk)):
                    return alpha_j
                if (alpha_hi - alpha_j) * curvature >= 0:
                    alpha_hi = alpha_lo
                alpha_lo = alpha_j
                f_tilde_lo = f_tilde_j
        return alpha_j
