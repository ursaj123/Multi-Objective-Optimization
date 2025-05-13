import numpy as np

class BK1:
    def __init__(self, n=2, lb=-5, ub=10, g_type=('zero', {})):
        r"""
        BK1 Problem

        f_1(x) = x_1^2 + x_2^2
        f_2(x) = (x_1 - 5)^2 + (x_2 - 5)^2

        Arguments:
        n: Number of variables (default 2)
        lb: Lower bound for variables
        ub: Upper bound for variables
        g_type: Type of g function (default 'zero')
            It can be one of ('zero', 'L1', 'indicator', 'max')
            Its parameters must be defined in the dictionary

        Defined Vars:
        bounds: Bounds for variables [(low, high), ...]
        constraints: List of constraints for the problem (default empty)
        """
        self.m = 2  # Number of objectives
        self.n = n
        self.lb = lb
        self.ub = ub
        self.bounds = tuple([(lb, ub) for _ in range(n)])
        self.constraints = []
        self.g_type = g_type
        self.true_pareto_front = self.calculate_optimal_pareto_front()
        self.ref_point = np.max(self.true_pareto_front, axis=0) + 1e-4

    def calculate_optimal_pareto_front(self, n_points=100):
        """
        Compute the true Pareto front points based on the parameterization:
        f_1 = 50 * (1 - lambda)^2
        f_2 = 50 * lambda^2
        for lambda in [0, 1]
        """
        lam = np.linspace(0, 1, n_points)
        f1 = 50 * (1 - lam)**2
        f2 = 50 * lam**2
        return np.stack([f1, f2], axis=1)

    def f1(self, x):
        """Objective f1: f1(x) = x_1^2 + x_2^2"""
        return np.sum(x**2)

    def grad_f1(self, x):
        """Gradient of f1: grad f1 = 2x"""
        return 2 * x

    def f2(self, x):
        """Objective f2: f2(x) = (x_1 - 5)^2 + (x_2 - 5)^2"""
        return np.sum((x - 5)**2)

    def grad_f2(self, x):
        """Gradient of f2: grad f2 = 2(x - 5)"""
        return 2 * (x - 5)

    def hess_f1(self, x):
        """Hessian of f1: hess f1 = 2 * I"""
        return 2 * np.eye(self.n)

    def hess_f2(self, x):
        """Hessian of f2: hess f2 = 2 * I"""
        return 2 * np.eye(self.n)

    def evaluate_f(self, x):
        """Evaluate vector of objective functions at x"""
        return np.array([self.f1(x), self.f2(x)])

    def evaluate_gradients_f(self, x):
        """Evaluate gradients of objectives at x"""
        return np.array([self.grad_f1(x), self.grad_f2(x)])

    def evaluate_hessians_f(self, x):
        """Evaluate Hessians of objectives at x"""
        return np.array([self.hess_f1(x), self.hess_f2(x)])

    def evaluate_g(self, z):
        """
        Evaluate vector of g_i(z) terms, depending on g_type
        """
        if self.g_type[0] == 'zero':
            return np.zeros(self.m)
        elif self.g_type[0] == 'L1':
            pass
        elif self.g_type[0] == 'indicator':
            pass
        else:
            pass

    def evaluate(self, x, z):
        """Evaluate full objective: F(x) + G(z)"""
        f_values = self.evaluate_f(x)
        g_values = self.evaluate_g(z)
        return f_values + g_values

    def f_list(self):
        """Return list of tuples (f_i, grad_f_i, hess_f_i)"""
        return [
            (self.f1, self.grad_f1, self.hess_f1),
            (self.f2, self.grad_f2, self.hess_f2)
        ]
