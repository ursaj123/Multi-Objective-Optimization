import numpy as np

class FF1:
    def __init__(self, g_type=('zero', {})):
        """
        FF1 Problem
        f_1(x_1, x_2) = 1 - exp(-(x_1 - 1)^2 - (x_2 + 1)^2)
        f_2(x_1, x_2) = 1 - exp(-(x_1 + 1)^2 - (x_2 - 1)^2)

        g_1(z) = zero/L1/indicator/max
        g_2(z) = zero/L1/indicator/max

        Arguments:
        g_type: Type of g function
                It can be one of ('zero', 'L1', 'indicator', 'max')
                Its parameters must be defined in the dictionary

        Defined Vars:
        bounds: Bounds for variables [(low, high), ...]
        constraints: List of constraints for the problem
        """
        self.m = 2  # Number of objectives
        self.n = 2  # Number of variables
        self.bounds = [(-5, 5), (-5, 5)]  #  Assume bounds for x1 and x2. You can adjust them.
        self.constraints = []
        self.g_type = g_type
        self.true_pareto_front = self.calculate_optimal_pareto_front()
        self.ref_point = np.max(self.true_pareto_front, axis=0) + 1e-4

        self.l1_ratios, self.l1_shifts = [], []
        if self.g_type[0]=='L1':
            for i in range(self.m):
                self.l1_ratios.append(1/((i+1)*self.m))
                self.l1_shifts.append(i)
            self.l1_ratios = np.array(self.l1_ratios)
            self.l1_shifts = np.array(self.l1_shifts)

    def calculate_optimal_pareto_front(self, num_points=100):
        """
        Approximates the Pareto front for FF1 by sampling x1 and x2 and
        filtering the non-dominated points.
        """
        x1_values = np.linspace(-3, 3, num_points) # sampling around (0,0)
        x2_values = np.linspace(-3, 3, num_points)
        X1, X2 = np.meshgrid(x1_values, x2_values)
        f1_values = self.f1(np.stack([X1.ravel(), X2.ravel()]))
        f2_values = self.f2(np.stack([X1.ravel(), X2.ravel()]))
        points = np.column_stack([f1_values, f2_values])

        # Pareto filtering (removing dominated points)
        is_pareto = np.ones(points.shape[0], dtype=bool)
        for i, c in enumerate(points):
            is_pareto[i] = np.all(np.any(points[:i] < c, axis=1)) and \
                           np.all(np.any(points[i+1:] < c, axis=1))
        return points[is_pareto]

    def f1(self, x):
        return 1 - np.exp(-(x[0] - 1)**2 - (x[1] + 1)**2)

    def grad_f1(self, x):
        return np.array([
            2 * (x[0] - 1) * np.exp(-(x[0] - 1)**2 - (x[1] + 1)**2),
            2 * (x[1] + 1) * np.exp(-(x[0] - 1)**2 - (x[1] + 1)**2)
        ])

    def f2(self, x):
        return 1 - np.exp(-(x[0] + 1)**2 - (x[1] - 1)**2)

    def grad_f2(self, x):
        return np.array([
            2 * (x[0] + 1) * np.exp(-(x[0] + 1)**2 - (x[1] - 1)**2),
            -2 * (x[1] - 1) * np.exp(-(x[0] + 1)**2 - (x[1] - 1)**2)
        ])

    def hess_f1(self, x):
        e = np.exp(-(x[0] - 1)**2 - (x[1] + 1)**2)
        return np.array([
            [2 * e - 4 * (x[0] - 1)**2 * e, -4 * (x[0] - 1) * (x[1] + 1) * e],
            [-4 * (x[0] - 1) * (x[1] + 1) * e, 2 * e - 4 * (x[1] + 1)**2 * e]
        ])

    def hess_f2(self, x):
        e = np.exp(-(x[0] + 1)**2 - (x[1] - 1)**2)
        return np.array([
            [2 * e - 4 * (x[0] + 1)**2 * e, 4 * (x[0] + 1) * (x[1] - 1) * e],
            [4 * (x[0] + 1) * (x[1] - 1) * e, 2 * e - 4 * (x[1] - 1)**2 * e]
        ])

    def evaluate_f(self, x):
        return np.array([self.f1(x), self.f2(x)])

    def evaluate_gradients_f(self, x):
        return np.array([self.grad_f1(x), self.grad_f2(x)])

    def evaluate_hessians_f(self, x):
        return np.array([self.hess_f1(x), self.hess_f2(x)])

    def evaluate_g(self, z):
        if self.g_type[0] == 'zero':
            return np.zeros(self.m)
        elif self.g_type[0] == 'L1':
            res = np.zeros(self.m)
            for i in range(self.m):
                res[i] = np.linalg.norm((z-self.l1_shifts[i])*self.l1_ratios[i], ord=1)
            return res
            pass
        elif self.g_type[0] == 'indicator':
            pass
        elif self.g_type[0] == 'max':
            pass
        return np.zeros(self.m)

    def evaluate(self, x, z):
        f_values = self.evaluate_f(x)
        g_values = self.evaluate_g(z)
        return f_values + g_values

    def f_list(self):
        return [
            (self.f1, self.grad_f1, self.hess_f1),
            (self.f2, self.grad_f2, self.hess_f2)
        ]