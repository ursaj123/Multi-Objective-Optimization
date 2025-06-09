import numpy as np

class BK1:
    def __init__(self, n=2, lb=-2, ub=2, g_type=('zero', {}), fact=1):
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
        self.bounds = [(lb, ub) for _ in range(n)]
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
            res = np.zeros(self.m)
            for i in range(self.m):
                res[i] = np.linalg.norm((z-self.l1_shifts[i])*self.l1_ratios[i], ord=1)
            return res
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
