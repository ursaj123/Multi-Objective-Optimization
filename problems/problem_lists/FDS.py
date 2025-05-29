import numpy as np

class FDS:
    def __init__(self, n=10, g_type=('zero', {})):
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
        self.true_pareto_front = self.calculate_optimal_pareto_front()
        self.ref_point = np.max(self.true_pareto_front, axis=0) + 1e-4

        self.l1_ratios, self.l1_shifts = [], []
        if self.g_type[0]=='L1':
            for i in range(self.m):
                self.l1_ratios.append(1/((i+1)*self.m))
                self.l1_shifts.append(i)
            self.l1_ratios = np.array(self.l1_ratios)
            self.l1_shifts = np.array(self.l1_shifts)

    def calculate_optimal_pareto_front(self, num_points=500):
        """
        Approximates the Pareto front for FDS.  This is complex due to the
        variable dimension.  We'll use a simplified sampling approach.
        """
        # Create sample points.  A more sophisticated sampling method
        # (like Latin Hypercube) would be better in higher dimensions.
        return np.array([1.0, 1.0, 1.0])  # Placeholder for the true Pareto front
        x_samples = np.linspace(-5, 5, num_points)  # Range for each x_k
        # Initialize an array to hold the objective values
        f_values = np.zeros((num_points*self.n, self.m))
        # Create a meshgrid to get all combinations of x_k values.
        if self.n == 1:
            X = [x_samples]
        elif self.n == 2:
            X1, X2 = np.meshgrid(x_samples, x_samples)
            X = [X1.ravel(), X2.ravel()]
        elif self.n == 3:
            X1, X2, X3 = np.meshgrid(x_samples, x_samples, x_samples)
            X = [X1.ravel(), X2.ravel(), X3.ravel()]
        elif self.n == 4:
            X1, X2, X3, X4 = np.meshgrid(x_samples, x_samples, x_samples, x_samples)
            X = [X1.ravel(), X2.ravel(), X3.ravel(), X4.ravel()]
        elif self.n > 4:
            # Placeholder: For n > 4, a full meshgrid is very inefficient.
            #  Consider using a different sampling strategy here.
            X = np.random.uniform(low=-5, high=5, size=(num_points, self.n))
            f_values = np.apply_along_axis(self.evaluate_f, 1, X)
            points = f_values
            is_pareto = np.ones(points.shape[0], dtype=bool)
            for i, c in enumerate(points):
                is_pareto[i] = np.all(np.any(points[:i] < c, axis=1)) and \
                               np.all(np.any(points[i+1:] < c, axis=1))
            return points[is_pareto]
        else:
          return np.array([])

        if self.n <= 4: #and X is not None:
            # Evaluate the objectives for each combination of x_k
            
            x_combinations = np.array(X).T
            print(x_combinations.shape)
            print(x_combinations[0])
            for i, x in enumerate(x_combinations):
                f_values[i] = self.evaluate_f(x)
            points = f_values
            # Pareto filtering (removing dominated points)
            is_pareto = np.ones(points.shape[0], dtype=bool)
            for i, c in enumerate(points):
                is_pareto[i] = np.all(np.any(points[:i] < c, axis=1)) and \
                               np.all(np.any(points[i+1:] < c, axis=1))
            return points[is_pareto]
        return np.array([])

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

