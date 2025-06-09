import numpy as np

class AP2:
    def __init__(self, lb=-2, ub=2, g_type=('zero', {}), fact=1):
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
        self.bounds = [(lb, ub) for _ in range(self.n)]
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
            res = np.zeros(self.m)
            for i in range(self.m):
                res[i] = np.linalg.norm((z-self.l1_shifts[i])*self.l1_ratios[i], ord=1)
            return res
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