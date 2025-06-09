import numpy as np

class IKK1:
    def __init__(self, g_type=('zero', {}), fact=1):
        """
        IKK1 Problem
        f_1(x_1) = x_1^2
        f_2(x_1) = (x_1 - 2)^2
        f_3(x_2) = x_2^2

        g_1(z) = zero/L1/indicator/max
        g_2(z) = zero/L1/indicator/max
        g_3(z) = zero/L1/indicator/max

        Arguments:
        g_type: Type of g function
                It can be one of ('zero', 'L1', 'indicator', 'max')
                Its parameters must be defined in the dictionary

        Defined Vars:
        bounds: Bounds for variables [(low, high), ...]
        constraints: List of constraints for the problem
        """
        self.m = 3  # Number of objectives
        self.n = 2  # Number of variables
        self.bounds = [(-2, 2), (-2, 2)]  # x1, x2
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
        return x[0]**2

    def grad_f1(self, x):
        return np.array([2 * x[0], 0])

    def f2(self, x):
        return (x[0] - 2)**2

    def grad_f2(self, x):
        return np.array([2 * (x[0] - 2), 0])

    def f3(self, x):
        return x[1]**2

    def grad_f3(self, x):
        return np.array([0, 2 * x[1]])

    def hess_f1(self, x):
        return np.array([[2, 0], [0, 0]])

    def hess_f2(self, x):
        return np.array([[2, 0], [0, 0]])

    def hess_f3(self, x):
        return np.array([[0, 0], [0, 2]])

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

