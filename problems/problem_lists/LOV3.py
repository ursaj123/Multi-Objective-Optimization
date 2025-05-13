import numpy as np

class LOV3:
    def __init__(self, n=2, lb=-5, ub=5, g_type=('zero', {})):
        r"""
        LOV3 Problem

        f_1(x) = -x[0]**2 - x[1]**2
        f_2(x) = -(x[0] - 6)**2 - (x[1] + 0.3)**2

        g_1(z) = zero/L1/indicator/max
        g_2(z) = zero/L1/indicator/max

        x and z are kept different for the sake of generality over algorithms.

        Arguments:
        m: Number of objectives (always 2 for LOV3)
        n: Number of variables (always 2 for LOV3)
        ub: Upper bound for variables
        g_type: Type of g function
                It can be one of ('zero', 'L1', 'indicator', 'max')
                Its parameters must be defined in the dictionary

        Defined Vars:
        bounds: Bounds for variables [(low, high), ...]

        constraints: List of constraints for the problem.
        """
        self.m = 2 # Number of objectives
        self.n = n
        self.lb = lb
        self.ub = ub
        self.bounds = tuple([(lb, ub) for _ in range(n)])
        self.constraints = []
        self.g_type = g_type
        self.true_pareto_front = self._calculate_optimal_pareto_front()
        self.ref_point = np.max(self.true_pareto_front, axis=0) + 1e-4

    def _calculate_optimal_pareto_front(self):
        """
        Sampling the Pareto front using a non-dominated sorting approach.
        """
        ndim = self.n
        n_samples = 500
        X = np.random.uniform(self.lb, self.ub, (n_samples, ndim))
        F = np.array([self._evaluate_f(x) for x in X])
        is_nondominated = np.ones(n_samples, dtype=bool)
        for i in range(n_samples):
            for j in range(n_samples):
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_nondominated[i] = False
                    break
        return np.unique(F[is_nondominated], axis=0)

    def f1(self, x):
        return -x[0]**2 - x[1]**2

    def grad_f1(self, x):
        return np.array([-2 * x[0], -2 * x[1]])

    def hess_f1(self, x):
        return np.array([[-2, 0], [0, -2]])

    def f2(self, x):
        return -(x[0] - 6)**2 - (x[1] + 0.3)**2

    def grad_f2(self, x):
        return np.array([-2 * (x[0] - 6), -2 * (x[1] + 0.3)])

    def hess_f2(self, x):
        return np.array([[-2, 0], [0, -2]])

    def _evaluate_f(self, x):
        return np.array([self.f1(x), self.f2(x)])

    def _evaluate_gradients_f(self, x):
        return np.array([self.grad_f1(x), self.grad_f2(x)])

    def _evaluate_hessians_f(self, x):
        return np.array([self.hess_f1(x), self.hess_f2(x)])

    def evaluate_g(self, z):
        if self.g_type[0] == 'zero':
            return np.zeros(self.m)
        elif self.g_type[0] == 'L1':
            w = self.g_type[1].get('w', np.ones(self.m))
            return np.sum(np.abs(z) * w)
        elif self.g_type[0] == 'indicator':
            domain_check = self.g_type[1].get('domain', lambda z: True)
            return np.zeros(self.m) if domain_check(z) else np.inf * np.ones(self.m)
        elif self.g_type[0] == 'max':
            return np.max(np.abs(z), axis=-1) * np.ones(self.m)
        else:
            raise ValueError(f"Unknown g_type: {self.g_type[0]}")

    def evaluate(self, x, z):
        f_values = self._evaluate_f(x)
        g_values = self.evaluate_g(z)
        return f_values + g_values

    def f_list(self):
        return [
            (self.f1, self.grad_f1, self.hess_f1),
            (self.f2, self.grad_f2, self.hess_f2)
        ]

