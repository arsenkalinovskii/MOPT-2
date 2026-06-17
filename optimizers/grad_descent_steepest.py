from optimizers.optimizer_base import Criteria
from optimizers.grad_descent_base import grad_descent_base
import scipy.optimize as opt


class grad_descent_steepest(grad_descent_base):
    def __init__(self, fun, grad_f, tol: float, starting_pos, linear_optimizer=opt.minimize_scalar,
                 criteria: Criteria = Criteria.NORM_GRAD,
                 max_iter: int = 1000, do_history: bool = False, interval_increase: float = 1):
        super().__init__(fun, grad_f, tol, starting_pos, "Steepest", criteria, max_iter, do_history)
        self.linear_optimizer = linear_optimizer
        self.interval_increase = interval_increase

    def calculate_alpha(self) -> float:
        gk = self.grad_cur_pos
        f_prev = self.f_cur_pos
        alpha_border = self.interval_increase
        border_delta = -gk * alpha_border
        while True:
            border_next = self.cur_pos + border_delta
            border_delta *= 2
            alpha_border *= 2
            f_next = self.calculate_function(border_next)
            if f_next >= f_prev:
                break
            f_prev = f_next

        fun = lambda t: self.calculate_function(self.cur_pos - t * gk)
        result = self.linear_optimizer(fun=fun, bracket=(0, alpha_border))
        return result.x
