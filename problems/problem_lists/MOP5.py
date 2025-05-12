import numpy as np

class MOP5:
    def __init__(self, g_type=('zero', {})):
        """
        MOP5 Problem

        f_1(x1, x2) = 0.5(x1^2 + x2^2) + sin(x1^2 + x2^2)
        f_2(x1, x2) = (3x1 - 2x2 + 4)^2 / 8 + (x1 - x2 + 1)^2 / 27 + 15
        f_3(x1, x2) = 1 / (x1^2 + x2^2 + 1) - 1.1 exp(-x1^2 - x2^2)

        Arguments:
        g_type: Type of g function
        """
        self.n = 2
        self.m = 3
        self.bounds = tuple([(-3, 3)] * 2)
        self.lb = -3
        self.ub = 3
        self.g_type = g_type
        self.constraints = []
        self.true_pareto_front = self.calculate_optimal_pareto_front()
        self.ref_point = np.max(self.true_pareto_front, axis=0) + 1e-4

    def f1(self, x):
        return 0.5 * (x[0]**2 + x[1]**2) + np.sin(x[0]**2 + x[1]**2)

    def grad_f1(self, x):
        return np.array([
            x[0] + 2 * x[0] * np.cos(x[0]**2 + x[1]**2),
            x[1] + 2 * x[1] * np.cos(x[0]**2 + x[1]**2)
        ])

    def hess_f1(self, x):
        H11 = 1 + 2 * np.cos(x[0]**2 + x[1]**2) - 4 * x[0]**2 * np.sin(x[0]**2 + x[1]**2)
        H12 = -4 * x[0] * x[1] * np.sin(x[0]**2 + x[1]**2)
        H21 = H12
        H22 = 1 + 2 * np.cos(x[0]**2 + x[1]**2) - 4 * x[1]**2 * np.sin(x[0]**2 + x[1]**2)
        return np.array([[H11, H12], [H21, H22]])


    def f2(self, x):
        return (3 * x[0] - 2 * x[1] + 4)**2 / 8 + (x[0] - x[1] + 1)**2 / 27 + 15

    def grad_f2(self, x):
        df2_dx1 = (2 * (3 * x[0] - 2 * x[1] + 4) * 3) / 8 + (2 * (x[0] - x[1] + 1)) / 27
        df2_dx2 = (2 * (3 * x[0] - 2 * x[1] + 4) * (-2)) / 8 + (2 * (x[0] - x[1] + 1) * (-1)) / 27
        return np.array([df2_dx1, df2_dx2])

    def hess_f2(self, x):
        h11 = 18/8 + 2/27
        h12 = -12/8 - 2/27
        h21 = h12
        h22 = 8/8 + 2/27
        return np.array([[h11, h12], [h21, h22]])


    def f3(self, x):
        return 1 / (x[0]**2 + x[1]**2 + 1) - 1.1 * np.exp(-x[0]**2 - x[1]**2)

    def grad_f3(self, x):
        df3_dx1 = -2 * x[0] / (x[0]**2 + x[1]**2 + 1)**2 + 2.2 * x[0] * np.exp(-x[0]**2 - x[1]**2)
        df3_dx2 = -2 * x[1] / (x[0]**2 + x[1]**2 + 1)**2 + 2.2 * x[1] * np.exp(-x[0]**2 - x[1]**2)
        return np.array([df3_dx1, df3_dx2])

    def hess_f3(self, x):
        H11 = (-2 / (x[0]**2 + x[1]**2 + 1)**2 + 8 * x[0]**2 / (x[0]**2 + x[1]**2 + 1)**3 +
               2.2 * np.exp(-x[0]**2 - x[1]**2) * (1 - 2 * x[0]**2))
        H12 = 8 * x[0] * x[1] / (x[0]**2 + x[1]**2 + 1)**3 - 4.4 * x[0] * x[1] * np.exp(-x[0]**2 - x[1]**2)
        H21 = H12
        H22 = (-2 / (x[0]**2 + x[1]**2 + 1)**2 + 8 * x[1]**2 / (x[0]**2 + x[1]**2 + 1)**3 +
               2.2 * np.exp(-x[0]**2 - x[1]**2) * (1 - 2 * x[1]**2))
        return np.array([[H11, H12], [H21, H22]])


    def evaluate_f(self, x):
        return np.array([self.f1(x), self.f2(x), self.f3(x)])

    def evaluate_gradients_f(self, x):
        return np.array([self.grad_f1(x), self.grad_f2(x), self.grad_f3(x)])

    def evaluate_hessians_f(self, x):
        return np.array([self.hess_f1(x), self.hess_f2(x), self.hess_f3(x)])

    def calculate_optimal_pareto_front(self, n_samples=100):
        #  Approximation of the Pareto front.
        x1_values = np.linspace(-1, 1, n_samples)
        x2_values = np.linspace(-1, 1, n_samples)
        pareto_front = np.zeros((n_samples, 3))
        for i in range(n_samples):
            x = np.array([x1_values[i], x2_values[i]])
            pareto_front[i, :] = self.evaluate_f(x)
        return pareto_front

    def evaluate_g(self, z):
        g_values = np.zeros(self.m)
        g_type, params = self.g_type
        if g_type == 'zero':
            return g_values
        elif g_type == 'L1':
            pass
        elif g_type == 'indicator':
            pass
        elif g_type == 'max':
            pass
        else:
            pass

    def evaluate(self, x, z):
        f_values = self.evaluate_f(x)
        g_values = self.evaluate_g(z)
        return f_values + g_values

    def f_list(self):
        return [
            (self.f1, self.grad_f1, self.hess_f1),
            (self.f2, self.grad_f2, self.hess_f2),
            (self.f3, self.grad_f3, self.hess_f3)
        ]
