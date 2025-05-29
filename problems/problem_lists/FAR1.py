import numpy as np

class FAR1:
    def __init__(self, g_type=('zero', {})):
        """
        FAR1 Problem
        f_1(x_1, x_2) = ... (complex exponential function)
        f_2(x_1, x_2) = ... (complex exponential function)

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
        self.bounds = [(-1, 1), (-1, 1)]  # x1, x2
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
        Approximates the Pareto front for FAR1 by sampling x1 and x2.
        """
        # print("Done")
        x1_values = np.linspace(-1, 1, num_points)
        x2_values = np.linspace(-1, 1, num_points)
        X1, X2 = np.meshgrid(x1_values, x2_values)
        # print("Done")
        f1_values = self.f1(np.stack([X1.ravel(), X2.ravel()]))
        print("Done")
        f2_values = self.f2(np.stack([X1.ravel(), X2.ravel()]))
        # print("Done")
        points = np.column_stack([f1_values, f2_values])

        # Pareto filtering (removing dominated points)
        is_pareto = np.ones(points.shape[0], dtype=bool)
        # print(points.shape)
        for i, c in enumerate(points):
            is_pareto[i] = np.all(np.any(points[:i] < c, axis=1)) and \
                           np.all(np.any(points[i+1:] < c, axis=1))

        # print("Done")
        return points[is_pareto]

    def f1(self, x):
        x1 = x[0]
        x2 = x[1]
        return -2 * np.exp(15 * (-(x1 - 0.1)**2 - x2**2)) - \
            np.exp(20 * (-(x1 - 0.6)**2 - (x2 - 0.6)**2)) + \
            np.exp(20 * (-(x1 + 0.6)**2 - (x2 - 0.6)**2)) + \
            np.exp(20 * (-(x1 - 0.6)**2 - (x2 + 0.6)**2)) + \
            np.exp(20 * (-(x1 + 0.6)**2 - (x2 + 0.6)**2))

    def grad_f1(self, x):
        x1 = x[0]
        x2 = x[1]
        df1_dx1 = -30 * (x1 - 0.1) * np.exp(15 * (-(x1 - 0.1)**2 - x2**2)) + \
            40 * (x1 - 0.6) * np.exp(20 * (-(x1 - 0.6)**2 - (x2 - 0.6)**2)) - \
            40 * (x1 + 0.6) * np.exp(20 * (-(x1 + 0.6)**2 - (x2 - 0.6)**2)) + \
            40 * (x1 - 0.6) * np.exp(20 * (-(x1 - 0.6)**2 - (x2 + 0.6)**2)) - \
            40 * (x1 + 0.6) * np.exp(20 * (-(x1 + 0.6)**2 - (x2 + 0.6)**2))
        df1_dx2 = -30 * x2 * np.exp(15 * (-(x1 - 0.1)**2 - x2**2)) + \
            40 * (x2 - 0.6) * np.exp(20 * (-(x1 - 0.6)**2 - (x2 - 0.6)**2)) - \
            40 * (x2 - 0.6) * np.exp(20 * (-(x1 + 0.6)**2 - (x2 - 0.6)**2)) + \
            40 * (x2 + 0.6) * np.exp(20 * (-(x1 - 0.6)**2 - (x2 + 0.6)**2)) + \
            40 * (x2 + 0.6) * np.exp(20 * (-(x1 + 0.6)**2 - (x2 + 0.6)**2))
        return np.array([df1_dx1, df1_dx2])

    def f2(self, x):
        x1 = x[0]
        x2 = x[1]
        return 2 * np.exp(20 * (-x1**2 - x2**2)) + \
            np.exp(20 * (-(x1 - 0.4)**2 - (x2 - 0.6)**2)) - \
            np.exp(20 * (-(x1 + 0.5)**2 - (x2 - 0.7)**2)) - \
            np.exp(20 * (-(x1 - 0.5)**2 - (x2 + 0.7)**2)) + \
            np.exp(20 * (-(x1 + 0.4)**2 - (x2 + 0.8)**2))

    def grad_f2(self, x):
        x1 = x[0]
        x2 = x[1]
        df2_dx1 = -40 * x1 * np.exp(20 * (-x1**2 - x2**2)) - \
            40 * (x1 - 0.4) * np.exp(20 * (-(x1 - 0.4)**2 - (x2 - 0.6)**2)) + \
            40 * (x1 + 0.5) * np.exp(20 * (-(x1 + 0.5)**2 - (x2 - 0.7)**2)) + \
            40 * (x1 - 0.5) * np.exp(20 * (-(x1 - 0.5)**2 - (x2 + 0.7)**2)) - \
            40 * (x1 + 0.4) * np.exp(20 * (-(x1 + 0.4)**2 - (x2 + 0.8)**2))

        df2_dx2 = -40 * x2 * np.exp(20 * (-x1**2 - x2**2)) - \
            40 * (x2 - 0.6) * np.exp(20 * (-(x1 - 0.4)**2 - (x2 - 0.6)**2)) + \
            40 * (x2 - 0.7) * np.exp(20 * (-(x1 + 0.5)**2 - (x2 - 0.7)**2)) + \
            40 * (x2 + 0.7) * np.exp(20 * (-(x1 - 0.5)**2 - (x2 + 0.7)**2)) - \
            40 * (x2 + 0.8) * np.exp(20 * (-(x1 + 0.4)**2 - (x2 + 0.8)**2))
        return np.array([df2_dx1, df2_dx2])

    def hess_f1(self, x):
        x1 = x[0]
        x2 = x[1]
        h11 = -30 * np.exp(15 * (-(x1 - 0.1)**2 - x2**2)) + 900 * (x1 - 0.1)**2 * np.exp(15 * (-(x1 - 0.1)**2 - x2**2)) + \
            40 * np.exp(20 * (-(x1 - 0.6)**2 - (x2 - 0.6)**2)) - 800 * (x1 - 0.6)**2 * np.exp(20 * (-(x1 - 0.6)**2 - (x2 - 0.6)**2)) + \
            40 * np.exp(20 * (-(x1 + 0.6)**2 - (x2 - 0.6)**2)) - 800 * (x1 + 0.6)**2 * np.exp(20 * (-(x1 + 0.6)**2 - (x2 - 0.6)**2)) + \
            40 * np.exp(20 * (-(x1 - 0.6)**2 - (x2 + 0.6)**2)) - 800 * (x1 - 0.6)**2 * np.exp(20 * (-(x1 - 0.6)**2 - (x2 + 0.6)**2)) + \
            40 * np.exp(20 * (-(x1 + 0.6)**2 - (x2 + 0.6)**2)) - 800 * (x1 + 0.6)**2 * np.exp(20 * (-(x1 + 0.6)**2 - (x2 + 0.6)**2))

        h12 = 900 * (x1 - 0.1) * x2 * np.exp(15 * (-(x1 - 0.1)**2 - x2**2)) + \
            800 * (x1 - 0.6) * (x2 - 0.6) * np.exp(20 * (-(x1 - 0.6)**2 - (x2 - 0.6)**2)) - \
            800 * (x1 + 0.6) * (x2 - 0.6) * np.exp(20 * (-(x1 + 0.6)**2 - (x2 - 0.6)**2)) + \
            800 * (x1 - 0.6) * (x2 + 0.6) * np.exp(20 * (-(x1 - 0.6)**2 - (x2 + 0.6)**2)) - \
            800 * (x1 + 0.6) * (x2 + 0.6) * np.exp(20 * (-(x1 + 0.6)**2 - (x2 + 0.6)**2))

        h21 = h12

        h22 = -30 * np.exp(15 * (-(x1 - 0.1)**2 - x2**2)) + 900 * x2**2 * np.exp(15 * (-(x1 - 0.1)**2 - x2**2)) + \
            40 * np.exp(20 * (-(x1 - 0.6)**2 - (x2 - 0.6)**2)) - 800 * (x2 - 0.6)**2 * np.exp(20 * (-(x1 - 0.6)**2 - (x2 - 0.6)**2)) + \
            40 * np.exp(20 * (-(x1 + 0.6)**2 - (x2 - 0.6)**2)) - 800 * (x2 - 0.6)**2 * np.exp(20 * (-(x1 + 0.6)**2 - (x2 - 0.6)**2)) + \
            40 * np.exp(20 * (-(x1 - 0.6)**2 - (x2 + 0.6)**2)) - 800 * (x2 + 0.6)**2 * np.exp(20 * (-(x1 - 0.6)**2 - (x2 + 0.6)**2)) + \
            40 * np.exp(20 * (-(x1 + 0.6)**2 - (x2 + 0.6)**2)) - 800 * (x2 + 0.6)**2 * np.exp(20 * (-(x1 + 0.6)**2 - (x2 + 0.6)**2))
        return np.array([[h11, h12], [h21, h22]])

    def hess_f2(self, x):
        x1 = x[0]
        x2 = x[1]

        h11 = -40 * np.exp(20 * (-x1**2 - x2**2)) + 800 * x1**2 * np.exp(20 * (-x1**2 - x2**2)) - \
            40 * np.exp(20 * (-(x1 - 0.4)**2 - (x2 - 0.6)**2)) + 800 * (x1 - 0.4)**2 * np.exp(20 * (-(x1 - 0.4)**2 - (x2 - 0.6)**2)) - \
            40 * np.exp(20 * (-(x1 + 0.5)**2 - (x2 - 0.7)**2)) + 800 * (x1 + 0.5)**2 * np.exp(20 * (-(x1 + 0.5)**2 - (x2 - 0.7)**2)) - \
            40 * np.exp(20 * (-(x1 - 0.5)**2 - (x2 + 0.7)**2)) + 800 * (x1 - 0.5)**2 * np.exp(20 * (-(x1 - 0.5)**2 - (x2 + 0.7)**2)) - \
            40 * np.exp(20 * (-(x1 + 0.4)**2 - (x2 + 0.8)**2)) + 800 * (x1 + 0.4)**2 * np.exp(20 * (-(x1 + 0.4)**2 - (x2 + 0.8)**2))

        h12 = 800 * x1 * x2 * np.exp(20 * (-x1**2 - x2**2)) + \
            800 * (x1 - 0.4) * (x2 - 0.6) * np.exp(20 * (-(x1 - 0.4)**2 - (x2 - 0.6)**2)) - \
            800 * (x1 + 0.5) * (x2 - 0.7) * np.exp(20 * (-(x1 + 0.5)**2 - (x2 - 0.7)**2)) - \
            800 * (x1 - 0.5) * (x2 + 0.7) * np.exp(20 * (-(x1 - 0.5)**2 - (x2 + 0.7)**2)) + \
            800 * (x1 + 0.4) * (x2 + 0.8) * np.exp(20 * (-(x1 + 0.4)**2 - (x2 + 0.8)**2))

        h21 = h12

        h22 = -40 * np.exp(20 * (-x1**2 - x2**2)) + 800 * x2**2 * np.exp(20 * (-x1**2 - x2**2)) - \
            40 * np.exp(20 * (-(x1 - 0.4)**2 - (x2- 0.6)**2)) + 800 * (x2 - 0.6)**2 * np.exp(20 * (-(x1 - 0.4)**2 - (x2 - 0.6)**2)) - \
            40 * np.exp(20 * (-(x1 + 0.5)**2 - (x2 - 0.7)**2)) + 800 * (x2 - 0.7)**2 * np.exp(20 * (-(x1 + 0.5)**2 - (x2 - 0.7)**2)) - \
            40 * np.exp(20 * (-(x1 - 0.5)**2 - (x2 + 0.7)**2)) + 800 * (x2 + 0.7)**2 * np.exp(20 * (-(x1 - 0.5)**2 - (x2 + 0.7)**2)) - \
            40 * np.exp(20 * (-(x1 + 0.4)**2 - (x2 + 0.8)**2)) + 800 * (x2 + 0.8)**2 * np.exp(20 * (-(x1 + 0.4)**2 - (x2 + 0.8)**2))
        return np.array([[h11, h12], [h21, h22]])

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
