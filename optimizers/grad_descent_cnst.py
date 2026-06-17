from optimizers.optimizer_base import Criteria
from optimizers.grad_descent_base import grad_descent_base


class grad_descent_cnst(grad_descent_base):
    def __init__(self, fun, grad_f, tol: float, starting_pos, alpha: float, criteria: Criteria = Criteria.NORM_GRAD,
                 max_iter: int = 1000, do_history: bool = False):
        super().__init__(fun, grad_f, tol, starting_pos, f"Constant step optimizer (step={alpha})", criteria, max_iter,
                         do_history)
        self.alpha = alpha

    def calculate_alpha(self) -> float:
        return self.alpha
