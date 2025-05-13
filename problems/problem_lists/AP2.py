import numpy as np

class AP2:
    def __init__(self, n=1, lb=-5, ub=5, g_type=('zero', {})):
        r"""
        AP2 Problem (One dimension)
        F_1(x) = x^2 - 4
        F_2(x) = (x - 1)^2

        Arguments:
        n: Number of variables (should be 1 for this problem)
        lb: Lower bound for variables
        ub: Upper bound for variables
        g_type: Type of g function
               It can be one of ('zero', 'L1', 'indicator', 'max')
               Its parameters must be defined in the dictionary

        Defined Vars:
        bounds: Bounds for variables [(low, high), ...]
        constraints: List of constraints for the problem
        """
        self.m = 2  # Number of objectives
        self.n = 1  # Fixed at 1 for this problem (one dimension)
        self.lb = lb
        self.ub = ub
        self.bounds = tuple([(lb, ub) for _ in range(self.n)])
        self.constraints = []
        self.g_type = g_type
        self.true_pareto_front = self.calculate_optimal_pareto_front()
        self.ref_point = np.max(self.true_pareto_front, axis=0) + 1e-4

    def calculate_optimal_pareto_front(self):
        """
        For AP2, the pareto front is the interval [0, 1]
        We'll sample a few points from this interval to represent the front
        """
        # Sample points in [0, 1]
        x_values = np.linspace(0, 1, 100)  # 100 points evenly spaced between 0 and 1
        
        # Evaluate the objectives at these points
        f_values = np.array([self.evaluate_f(np.array([x])) for x in x_values])
        
        return f_values

    def f1(self, x):
        """F_1(x) = x^2 - 4"""
        return x[0]**2 - 4

    def grad_f1(self, x):
        """Gradient of F_1 with respect to x"""
        return np.array([2 * x[0]])

    def hess_f1(self, x):
        """Hessian of F_1 with respect to x"""
        return np.array([[2]])

    def f2(self, x):
        """F_2(x) = (x - 1)^2"""
        return (x[0] - 1)**2

    def grad_f2(self, x):
        """Gradient of F_2 with respect to x"""
        return np.array([2 * (x[0] - 1)])

    def hess_f2(self, x):
        """Hessian of F_2 with respect to x"""
        return np.array([[2]])

    def evaluate_f(self, x):
        """Evaluate all objective functions at point x"""
        return np.array([self.f1(x), self.f2(x)])

    def evaluate_gradients_f(self, x):
        """Evaluate gradients of all objective functions at point x"""
        return np.array([self.grad_f1(x), self.grad_f2(x)])

    def evaluate_hessians_f(self, x):
        """Evaluate Hessians of all objective functions at point x"""
        return np.array([self.hess_f1(x), self.hess_f2(x)])

    def evaluate_g(self, z):
        """Evaluate the g functions based on the specified type"""
        if self.g_type[0] == 'zero':
            return np.zeros(self.m)
        elif self.g_type[0] == 'L1':
            # Implement L1 regularization when needed
            # This would depend on parameters in self.g_type[1]
            pass
        elif self.g_type[0] == 'indicator':
            # Implement indicator function when needed
            pass
        else:  # 'max' case
            # Implement max function when needed
            pass

    def evaluate(self, x, z):
        """Evaluate F(x) + G(z)"""
        f_values = self.evaluate_f(x)
        g_values = self.evaluate_g(z)
        return f_values + g_values

    def f_list(self):
        """Return a list of tuples containing functions, gradients, and hessians"""
        return [
            (self.f1, self.grad_f1, self.hess_f1),
            (self.f2, self.grad_f2, self.hess_f2)
        ]