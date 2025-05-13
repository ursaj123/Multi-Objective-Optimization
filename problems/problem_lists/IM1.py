import numpy as np

class IM1:
    def __init__(self, g_type=('zero', {})):
        """
        IM1 Problem
        f_1(x_1) = 2 * sqrt(x_1)
        f_2(x_1, x_2) = x_1 * (1 - x_2) + 5

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
        self.bounds = [(1, 4), (1, 2)]  # x1, x2
        self.constraints = []
        self.g_type = g_type
        self.true_pareto_front = self.calculate_optimal_pareto_front()
        self.ref_point = np.max(self.true_pareto_front, axis=0) + 1e-4

    def calculate_optimal_pareto_front(self, num_points=100):
        """
        Approximates the Pareto front for IM1.  The Pareto front is a curve
        obtained by setting x2 = 2 and varying x1.
        """
        x1_values = np.linspace(1, 4, num_points)
        f1_values = 2 * np.sqrt(x1_values)
        f2_values = 5 - x1_values  # x2 = 2
        pareto_front = np.column_stack([f1_values, f2_values])
        return pareto_front

    def f1(self, x):
        return 2 * np.sqrt(x[0])

    def grad_f1(self, x):
        return np.array([1 / np.sqrt(x[0]), 0])

    def f2(self, x):
        return x[0] * (1 - x[1]) + 5

    def grad_f2(self, x):
        return np.array([1 - x[1], -x[0]])

    def hess_f1(self, x):
        return np.array([[-0.5 * x[0]**(-1.5), 0], [0, 0]])

    def hess_f2(self, x):
        return np.array([[0, -1], [-1, 0]])

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
