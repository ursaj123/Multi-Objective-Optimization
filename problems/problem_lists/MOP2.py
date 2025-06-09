import numpy as np

class MOP2:
    def __init__(self, n=2, g_type=('zero', {}), fact=1):
        """
        MOP2 Problem

        f_1(x) = 1 - exp(-sum((xi - 1/sqrt(n))^2))
        f_2(x) = 1 - exp(-sum((xi + 1/sqrt(n))^2))

        Arguments:
        n: Number of variables
        g_type: Type of g function
        """
        self.m = 2  # Number of objectives
        self.n = n
        self.lb = -2
        self.ub = 2
        self.bounds = [(self.lb, self.ub) for _ in range(n)]
        self.g_type = g_type
        self.constraints = []
        

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
        return 1 - np.exp(-np.sum((x - 1/np.sqrt(self.n))**2))

    def grad_f1(self, x):
        exponent = -np.sum((x - 1/np.sqrt(self.n))**2)
        exponential_term = np.exp(exponent)
        gradient = 2 * (x - 1/np.sqrt(self.n)) * exponential_term
        return gradient

    def hess_f1(self, x):
        exponent = -np.sum((x - 1/np.sqrt(self.n))**2)
        exponential_term = np.exp(exponent)
        identity_matrix = np.eye(self.n)
        outer_product = 2 * np.outer((x - 1/np.sqrt(self.n)), (x - 1/np.sqrt(self.n)))
        hessian = 2 * (exponential_term * identity_matrix + 2 * exponential_term * outer_product)
        return hessian


    def f2(self, x):
        return 1 - np.exp(-np.sum((x + 1/np.sqrt(self.n))**2))

    def grad_f2(self, x):
        exponent = -np.sum((x + 1/np.sqrt(self.n))**2)
        exponential_term = np.exp(exponent)
        gradient = 2 * (x + 1/np.sqrt(self.n)) * exponential_term
        return gradient

    def hess_f2(self, x):
        exponent = -np.sum((x + 1/np.sqrt(self.n))**2)
        exponential_term = np.exp(exponent)
        identity_matrix = np.eye(self.n)
        outer_product = 2 * np.outer((x + 1/np.sqrt(self.n)), (x + 1/np.sqrt(self.n)))
        hessian = 2 * (exponential_term * identity_matrix + 2 * exponential_term * outer_product)
        return hessian

    def evaluate_f(self, x):
        return np.array([self.f1(x), self.f2(x)])

    def evaluate_gradients_f(self, x):
        return np.array([self.grad_f1(x), self.grad_f2(x)])

    def evaluate_hessians_f(self, x):
        return np.array([self.hess_f1(x), self.hess_f2(x)])

    def calculate_optimal_pareto_front(self, n_samples=100):
        # The Pareto front for MOP2 lies between [0, 1] x [0, 1].
        # We can sample points along this front.  A simple way is to vary
        # the input x_i values between -1/sqrt(n) and 1/sqrt(n).
        x_values = np.linspace(-1/np.sqrt(self.n), 1/np.sqrt(self.n), n_samples)
        pareto_front = np.zeros((n_samples, 2))
        for i in range(n_samples):
            x = np.full(self.n, x_values[i])  # Create an n-dimensional vector
            pareto_front[i, :] = self.evaluate_f(x)
        return pareto_front

    def evaluate_g(self, z):
        g_values = np.zeros(self.m)
        g_type, params = self.g_type
        if g_type == 'zero':
            return g_values
        elif g_type == 'L1':
            res = np.zeros(self.m)
            for i in range(self.m):
                res[i] = np.linalg.norm((z-self.l1_shifts[i])*self.l1_ratios[i], ord=1)
            return res
        elif g_type == 'indicator':
            pass
        elif g_type == 'max':
            pass

    def evaluate(self, x, z):
        f_values = self.evaluate_f(x)
        g_values = self.evaluate_g(z)
        return f_values + g_values

    def f_list(self):
        return [
            (self.f1, self.grad_f1, self.hess_f1),
            (self.f2, self.grad_f2, self.hess_f2)
        ]


