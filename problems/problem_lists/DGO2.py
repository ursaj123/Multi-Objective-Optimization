import numpy as np

class DGO2:
    def __init__(self, lb=-9, ub=9, g_type=('zero', {})):
        """
        DGO2 Problem
        f_1(x) = x^2
        f_2(x) = 9 - sqrt(81 - x^2)

        g_1(z) = zero/L1/indicator/max
        g_2(z) = zero/L1/indicator/max

        Arguments:
        ub: Upper bound for the variable
        lb: Lower bound for the variable
        g_type: Type of g function
                It can be one of ('zero', 'L1', 'indicator', 'max')
                Its parameters must be defined in the dictionary

        Defined Vars:
        bounds: Bounds for the variable [(low, high)]
        constraints: List of constraints (empty for this problem)
        """
        self.m = 2  # Number of objectives
        self.n = 1  # Number of variables
        self.lb = lb
        self.ub = ub
        self.bounds = tuple([(lb, ub)])
        self.constraints = []
        self.g_type = g_type
        self.true_pareto_front = self.calculate_optimal_pareto_front()
        self.ref_point = np.max(self.true_pareto_front, axis=0) + 1e-4

    def calculate_optimal_pareto_front(self, num_points=100):
        """
        Generates points on the Pareto front.
        """
        x_values = np.linspace(0, self.ub, num_points) # Considering x >= 0 for the Pareto front
        f1_values = x_values**2
        f2_values = 9 - np.sqrt(81 - x_values**2)
        return np.column_stack([f1_values, f2_values])

    def f1(self, x):
        return x**2

    def grad_f1(self, x):
        return 2 * x

    def f2(self, x):
        return 9 - np.sqrt(81 - x**2)

    def grad_f2(self, x):
        return x / np.sqrt(81 - x**2) if (81 - x**2) > 0 else np.inf

    def hess_f1(self, x):
        return 2 * np.ones_like(x)

    def hess_f2(self, x):
        return 81 / (np.sqrt(81 - x**2)**3) if (81 - x**2) > 0 else np.inf

    def evaluate_f(self, x):
        return np.array([self.f1(x), self.f2(x)])

    def evaluate_gradients_f(self, x):
        grad_f2_val = self.grad_f2(x)
        return np.array([self.grad_f1(x), grad_f2_val])

    def evaluate_hessians_f(self, x):
        hess_f2_val = self.hess_f2(x)
        return np.array([self.hess_f1(x), hess_f2_val])

    def evaluate_g(self, z):
        if self.g_type[0] == 'zero':
            return np.zeros(self.m)
        elif self.g_type[0] == 'L1':
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

