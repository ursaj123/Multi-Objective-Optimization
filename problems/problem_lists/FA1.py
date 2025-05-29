import numpy as np

class FA1:
    def __init__(self, g_type=('zero', {})):
        """
        FA1 Problem
        f_1(x_1) = (1 - exp(-4x_1)) / (1 - exp(-4))
        f_2(x_1, x_2, x_3) = (x_2 + 1) * (1 - sqrt(f_1(x_1) / (x_2 + 1)))
        f_3(x_1, x_2, x_3) = (x_3 + 1) * (1 - (f_1(x_1) / (x_3 + 1))**0.1)

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
        self.n = 3  # Number of variables
        # self.bounds = [(0, 1), (-np.inf, np.inf), (-np.inf, np.inf)]  # x1, x2, x3
        self.bounds = [(0, 1), (-5, 5), (-5, 5)]  # x1, x2, x3
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
        Approximates the Pareto front for FA1.  Due to the unconstrained nature
        of x2 and x3, we'll sample x1 and then find corresponding f2, f3
        by maximizing x2, x3.  In practice, maximizing x2 and x3 means
        setting them to a very large value.
        """
        x1_values = np.linspace(0, 1, num_points)
        f1_values = (1 - np.exp(-4 * x1_values)) / (1 - np.exp(-4))
        x2_max = 1e10  # A large value to approximate "infinity"
        x3_max = 1e10
        f2_values = (x2_max + 1) * (1 - np.sqrt(f1_values / (x2_max + 1)))
        f3_values = (x3_max + 1) * (1 - (f1_values / (x3_max + 1))**0.1)
        pareto_front = np.column_stack([f1_values, f2_values, f3_values])

        # In this specific problem, all generated points are non-dominated
        return pareto_front

    def f1(self, x):
        return (1 - np.exp(-4 * x[0])) / (1 - np.exp(-4))

    def grad_f1(self, x):
        return (4 * np.exp(-4 * x[0])) / (1 - np.exp(-4))

    def f2(self, x):
        return (x[1] + 1) * (1 - np.sqrt(self.f1(x) / (x[1] + 1)))

    def grad_f2(self, x):
        df1 = self.grad_f1(x)
        return np.array([
            -0.5 * (x[1] + 1) * (self.f1(x) / (x[1] + 1))**(-0.5) * (df1 / (x[1] + 1)),
            1 - np.sqrt(self.f1(x) / (x[1] + 1)) + 0.5 * np.sqrt(self.f1(x) / (x[1] + 1))
            , 0
        ])

    def f3(self, x):
        return (x[2] + 1) * (1 - (self.f1(x) / (x[2] + 1))**0.1)

    def grad_f3(self, x):
        df1 = self.grad_f1(x)
        return np.array([
            -0.1 * (x[2] + 1) * (self.f1(x) / (x[2] + 1))**(-0.9) * (df1 / (x[2] + 1)),
            0,
            1 - (self.f1(x) / (x[2] + 1))**0.1 + 0.1 * (self.f1(x) / (x[2] + 1))**0.1
        ])

    def hess_f1(self, x):
        return np.array([[-16 * np.exp(-4 * x[0]) / (1 - np.exp(-4)), 0, 0],
                         [0, 0, 0],
                         [0, 0, 0]])

    def hess_f2(self, x):
        df1 = self.grad_f1(x)
        d2f1 = -16 * np.exp(-4 * x[0]) / (1 - np.exp(-4))
        return np.array([
            [-0.25 * (x[1] + 1)**(-1.5) * self.f1(x)**(-0.5) * df1**2
             - 0.5 * (x[1] + 1)**(-0.5) * self.f1(x)**(-1.5) * (x[1]+1)**(-1) * d2f1,
             0.5 * (x[1] + 1)**(-1.5) * self.f1(x)**(-0.5) * df1, 0],
            [0.5 * (x[1] + 1)**(-1.5) * self.f1(x)**(-0.5) * df1,
             0.25 * (x[1] + 1)**(-2.5) * self.f1(x)**0.5, 0],
            [0, 0, 0]
        ])

    def hess_f3(self, x):
        df1 = self.grad_f1(x)
        d2f1 = -16 * np.exp(-4 * x[0]) / (1 - np.exp(-4))
        return np.array([
            [-0.01 * (x[2] + 1)**(-1.9) * self.f1(x)**(-0.9) * df1**2
             - 0.1 * (x[2] + 1)**(-0.9) * self.f1(x)**(-1.9) * (x[2] + 1)**(-1) * d2f1,
             0, 0],
            [0, 0, 0],
            [0, 0,
             0.01 * (x[2] + 1)**(-2.1) * self.f1(x)**0.1]
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
