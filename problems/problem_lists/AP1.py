import numpy as np

class AP1:
    def __init__(self, n=2, lb=-2, ub=2, g_type=('zero', {}), fact=1):
        r"""
        AP1 Problem
        F_1(x_1, x_2) = (1/4)[(x_1-1)^4 + 2(x_2-2)^4]
        F_2(x_1, x_2) = e^((x_1+x_2)/2) + x_1^2 + x_2^2
        F_3(x_1, x_2) = (1/6)(e^(-x_1) + 2e^(-x_2))

        Arguments:
        n: Number of variables (should be 2 for this problem)
        lb: Lower bound for variables
        ub: Upper bound for variables
        g_type: Type of g function
               It can be one of ('zero', 'L1', 'indicator', 'max')
               Its parameters must be defined in the dictionary

        Defined Vars:
        bounds: Bounds for variables [(low, high), ...]
        constraints: List of constraints for the problem
        """
        self.m = 3  # Number of objectives
        self.n = 2  # Fixed at 2 for this problem
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
        """F_1(x_1, x_2) = (1/4)[(x_1-1)^4 + 2(x_2-2)^4]"""
        return 0.25 * ((x[0] - 1)**4 + 2 * (x[1] - 2)**4)

    def grad_f1(self, x):
        """Gradient of F_1 with respect to x"""
        dx1 = (x[0] - 1)**3  # Derivative of (x_1-1)^4 with respect to x_1
        dx2 = 2 * (x[1] - 2)**3  # Derivative of 2(x_2-2)^4 with respect to x_2
        return np.array([dx1, dx2])

    def hess_f1(self, x):
        """Hessian of F_1 with respect to x"""
        h11 = 3 * (x[0] - 1)**2  # Second derivative with respect to x_1
        h22 = 6 * (x[1] - 2)**2  # Second derivative with respect to x_2
        h12 = 0  # Mixed partial derivative
        return np.array([[h11, h12], [h12, h22]])

    def f2(self, x):
        """F_2(x_1, x_2) = e^((x_1+x_2)/2) + x_1^2 + x_2^2"""
        return np.exp((x[0] + x[1]) / 2) + x[0]**2 + x[1]**2

    def grad_f2(self, x):
        """Gradient of F_2 with respect to x"""
        exp_term = np.exp((x[0] + x[1]) / 2)
        dx1 = 0.5 * exp_term + 2 * x[0]  # Derivative with respect to x_1
        dx2 = 0.5 * exp_term + 2 * x[1]  # Derivative with respect to x_2
        return np.array([dx1, dx2])

    def hess_f2(self, x):
        """Hessian of F_2 with respect to x"""
        exp_term = np.exp((x[0] + x[1]) / 2)
        h11 = 0.25 * exp_term + 2  # Second derivative with respect to x_1
        h22 = 0.25 * exp_term + 2  # Second derivative with respect to x_2
        h12 = 0.25 * exp_term  # Mixed partial derivative
        return np.array([[h11, h12], [h12, h22]])

    def f3(self, x):
        """F_3(x_1, x_2) = (1/6)(e^(-x_1) + 2e^(-x_2))"""
        return (1/6) * (np.exp(-x[0]) + 2 * np.exp(-x[1]))

    def grad_f3(self, x):
        """Gradient of F_3 with respect to x"""
        dx1 = -(1/6) * np.exp(-x[0])  # Derivative with respect to x_1
        dx2 = -(1/3) * np.exp(-x[1])  # Derivative with respect to x_2
        return np.array([dx1, dx2])

    def hess_f3(self, x):
        """Hessian of F_3 with respect to x"""
        h11 = (1/6) * np.exp(-x[0])  # Second derivative with respect to x_1
        h22 = (1/3) * np.exp(-x[1])  # Second derivative with respect to x_2
        h12 = 0  # Mixed partial derivative
        return np.array([[h11, h12], [h12, h22]])

    def evaluate_f(self, x):
        """Evaluate all objective functions at point x"""
        return np.array([self.f1(x), self.f2(x), self.f3(x)])

    def evaluate_gradients_f(self, x):
        """Evaluate gradients of all objective functions at point x"""
        return np.array([self.grad_f1(x), self.grad_f2(x), self.grad_f3(x)])

    def evaluate_hessians_f(self, x):
        """Evaluate Hessians of all objective functions at point x"""
        return np.array([self.hess_f1(x), self.hess_f2(x), self.hess_f3(x)])

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
            (self.f2, self.grad_f2, self.hess_f2),
            (self.f3, self.grad_f3, self.hess_f3)
        ]