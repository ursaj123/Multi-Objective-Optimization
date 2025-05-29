import numpy as np

class AP3:
    def __init__(self, n=2, lb=-5, ub=5, g_type=('zero', {})):
        r"""
        AP3 Problem (Two dimension)
        F_1(x_1, x_2) = (1/4)[(x_1-1)^4 + 2(x_2-2)^4]
        F_2(x_1, x_2) = (x_2 - x_1^2)^2 + (1-x_1)^2

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
        self.m = 2  # Number of objectives
        self.n = 2  # Fixed at 2 for this problem
        self.lb = lb
        self.ub = ub
        self.bounds = [(lb, ub) for _ in range(self.n)]
        self.constraints = []
        self.g_type = g_type
        self.true_pareto_front = self.calculate_optimal_pareto_front()
        self.ref_point = np.max(self.true_pareto_front, axis=0) + 1e-4

        self.l1_ratios, self.l1_shifts = [], []
        if self.g_type[0] == 'L1':
            for i in range(self.m):
                self.l1_ratios.append(1 / ((i + 1) * self.m))
                self.l1_shifts.append(i)
            self.l1_ratios = np.array(self.l1_ratios)
            self.l1_shifts = np.array(self.l1_shifts)

    def calculate_optimal_pareto_front(self):
        """
        For AP3, we only know one point on the Pareto front: x* = (1, 2)
        F₂ is non-convex, so the complete Pareto front would require more analysis
        """
        # Evaluate the objectives at the known Pareto optimal point (1, 2)
        opt_point = np.array([1.0, 2.0])
        f_values = self.evaluate_f(opt_point)
        
        # Since we only have one confirmed point, return it as an array
        return np.array([f_values])

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
        """F_2(x_1, x_2) = (x_2 - x_1^2)^2 + (1-x_1)^2"""
        return (x[1] - x[0]**2)**2 + (1 - x[0])**2

    def grad_f2(self, x):
        """Gradient of F_2 with respect to x"""
        # For F_2(x) = (x_2 - x_1^2)^2 + (1-x_1)^2
        dx1 = -4 * (x[1] - x[0]**2) * x[0] - 2 * (1 - x[0])
        dx2 = 2 * (x[1] - x[0]**2)
        return np.array([dx1, dx2])

    def hess_f2(self, x):
        """Hessian of F_2 with respect to x"""
        # For F_2(x) = (x_2 - x_1^2)^2 + (1-x_1)^2
        # Partial derivatives: ∂²F/∂x_1², ∂²F/∂x_1∂x_2, ∂²F/∂x_2∂x_1, ∂²F/∂x_2²
        h11 = 12 * x[0]**2 - 4 * x[1] + 2  # ∂²F/∂x_1²
        h12 = -4 * x[0]  # ∂²F/∂x_1∂x_2
        h21 = -4 * x[0]  # ∂²F/∂x_2∂x_1
        h22 = 2  # ∂²F/∂x_2²
        return np.array([[h11, h12], [h21, h22]])

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