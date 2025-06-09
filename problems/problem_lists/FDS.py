import numpy as np

class FDS:
    def __init__(self, n=2, g_type=('zero', {}), fact=1):
        """
        Correct its pareto front code

        
        FDS Problem (variable dimension)
        F_1(x) = (1 / n^2) * sum(k * (x_k - k)^4) for k = 1 to n
        F_2(x) = exp((1 / n) * sum(x_k)) + ||x||^2
        F_3(x) = (1 / (n * (n + 1))) * sum(k * (n - k + 1) * exp(-x_k)) for k = 1 to n

        Arguments:
        n: Number of variables
        g_type: Type of g function
                It can be one of ('zero', 'L1', 'indicator', 'max')
                Its parameters must be defined in the dictionary

        Defined Vars:
        n: Number of variables
        bounds: Bounds for variables [(low, high), ...]
        constraints: List of constraints for the problem
        """
        self.m = 3  # Number of objectives
        self.n = n  # Number of variables
        self.bounds = [(-1, 1)] * n  # Default bounds, you might need to adjust
        self.constraints = []
        self.g_type = g_type

        self.l1_ratios, self.l1_shifts = [], []
        if self.g_type[0]=='L1':
            for i in range(self.m):
                self.l1_ratios.append(1/((i+1)*self.n*fact))
                self.l1_shifts.append(i)
            self.l1_ratios = np.array(self.l1_ratios)
            self.l1_shifts = np.array(self.l1_shifts)

    def feasible_space(self):
        test_x = np.random.uniform(self.bounds[0][0], self.bounds[0][1], size=(50000, self.n))
        f_values = np.array([self.evaluate(x, x) for x in test_x])
        return f_values

    def f1(self, x):
        n = self.n
        return (1 / n**2) * np.sum([k * (x[k-1] - k)**4 for k in range(1, n + 1)])

    def grad_f1(self, x):
        n = self.n
        return np.array([(4 / n**2) * k * (x[k-1] - k)**3 for k in range(1, n + 1)])

    def f2(self, x):
        n = self.n
        sum_xk = np.sum(x)
        norm_x_sq = np.sum(x**2)
        return np.exp(sum_xk / n) + norm_x_sq

    def grad_f2(self, x):
        n = self.n
        exp_term = np.exp(np.sum(x) / n)
        return np.array([exp_term / n + 2 * x[k-1] for k in range(1, n + 1)])

    def f3(self, x):
        n = self.n
        return (1 / (n * (n + 1))) * np.sum([k * (n - k + 1) * np.exp(-x[k-1]) for k in range(1, n + 1)])

    def grad_f3(self, x):
        n = self.n
        return np.array([
            -(1 / (n * (n + 1))) * k * (n - k + 1) * np.exp(-x[k-1]) for k in range(1, n + 1)
        ])

    def hess_f1(self, x):
        n = self.n
        return np.diag([(12 / n**2) * k * (x[k-1] - k)**2 for k in range(1, n + 1)])

    def hess_f2(self, x):
        n = self.n
        exp_term = np.exp(np.sum(x) / n)
        H = np.full((n, n), exp_term / n**2)
        H += np.diag([2] * n)
        return H

    def hess_f3(self, x):
        n = self.n
        return np.diag([
            (1 / (n * (n + 1))) * k * (n - k + 1) * np.exp(-x[k-1]) for k in range(1, n + 1)
        ])

    def evaluate_f(self, x):
        return np.array([self.f1(x), self.f2(x), self.f3(x)])

    def evaluate_gradients_f(self, x):
        return np.array([self.grad_f1(x), self.grad_f2(x), self.grad_f3(x)])

    def evaluate_hessians_f(self, x):
        return np.array([self.hess_f1(x), self.hess_f2(x), self.hess_f3(x)])

    def evaluate_g(self, z):
        if self.g_type[0] == 'zero':
            return np.zeros(self.m)
        elif self.g_type[0] == 'L1':
            res = np.zeros(self.m)
            for i in range(self.m):
                res[i] = np.linalg.norm((z-self.l1_shifts[i])*self.l1_ratios[i], ord=1)
            return res
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
            (self.f2, self.grad_f2, self.hess_f2),
            (self.f3, self.grad_f3, self.hess_f3)
        ]

