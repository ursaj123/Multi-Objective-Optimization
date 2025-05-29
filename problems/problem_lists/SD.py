import numpy as np

class SD:
    def __init__(self, n=4, g_type=('zero', {}), num_pf_samples=100):
        r"""
        SD Problem

        Objective Functions:
        f1(x) = 2*x1 + sqrt(2)*x2 + sqrt(2)*x3 + x4
        f2(x) = 2/x1 + 2*sqrt(2)/x2 + 2*sqrt(2)/x3 + x4

        Variables: x = (x1, x2, x3, x4)^T
        Bounds:
        x1 in [1, 3]
        x2 in [sqrt(2), 3]
        x3 in [sqrt(2), 3]
        x4 in [1, 3]
        """
        self.m = 2  # Number of objectives
        if n != 4:
            raise ValueError("SD problem is defined for n=4 variables.")
        self.n = n
        
        self.lb = np.array([1.0, np.sqrt(2), np.sqrt(2), 1.0])
        self.ub = np.array([3.0, 3.0, 3.0, 3.0])
        
        self.bounds = [(self.lb[i], self.ub[i]) for i in range(self.n)]
        self.constraints = [] # No explicit constraints other than bounds
        self.g_type = g_type # For the G(z) part of the optimization problem F(x) + G(z)
        
        self.num_pf_samples = num_pf_samples # Used in calculate_optimal_pareto_front
        self.true_pareto_front = self.calculate_optimal_pareto_front()
        
        if hasattr(self, 'true_pareto_front') and self.true_pareto_front.size > 0:
            self.ref_point = np.max(self.true_pareto_front, axis=0) + 1e-4
        else:
             # Fallback reference point if Pareto front calculation is empty
             # Based on PF endpoints: f1_max approx 13.73, f2_max approx 7
             print("Warning: SD Pareto front sampling might be empty. Using a generic ref point.")
             self.ref_point = np.array([14.0, 7.5]) 

        self.l1_ratios, self.l1_shifts = [], []
        if self.g_type[0]=='L1':
            for i in range(self.m):
                self.l1_ratios.append(1/((i+1)*self.m))
                self.l1_shifts.append(i)
            self.l1_ratios = np.array(self.l1_ratios)
            self.l1_shifts = np.array(self.l1_shifts)

    def f1(self, x_vars):
        """
        Objective function f1(x) = 2*x1 + sqrt(2)*x2 + sqrt(2)*x3 + x4
        """
        x = np.asarray(x_vars)
        return 2*x[0] + np.sqrt(2)*x[1] + np.sqrt(2)*x[2] + x[3]

    def grad_f1(self, x_vars):
        """
        Gradient of f1(x).
        grad_f1 = [2, sqrt(2), sqrt(2), 1]^T
        """
        # x_vars is not used as the gradient is constant.
        # Included for API consistency.
        return np.array([2.0, np.sqrt(2), np.sqrt(2), 1.0])

    def hess_f1(self, x_vars):
        """
        Hessian of f1(x).
        H_f1 = Zero matrix (4x4)
        """
        # x_vars is not used as the Hessian is constant.
        return np.zeros((self.n, self.n))

    def f2(self, x_vars):
        """
        Objective function f2(x) = 2/x1 + 2*sqrt(2)/x2 + 2*sqrt(2)/x3 + x4
        """
        x = np.asarray(x_vars)
        # Add small epsilon to denominators to prevent division by zero if x_i are ever outside bounds
        # or exactly zero, though bounds should prevent this.
        epsilon = 1e-9
        term1 = 2.0 / (x[0] + epsilon)
        term2 = (2.0 * np.sqrt(2)) / (x[1] + epsilon)
        term3 = (2.0 * np.sqrt(2)) / (x[2] + epsilon)
        return term1 + term2 + term3 + x[3]

    def grad_f2(self, x_vars):
        """
        Gradient of f2(x).
        grad_f2 = [-2/x1^2, -2*sqrt(2)/x2^2, -2*sqrt(2)/x3^2, 1]^T
        """
        x = np.asarray(x_vars)
        epsilon = 1e-9
        grad = np.zeros(self.n)
        grad[0] = -2.0 / (x[0]**2 + epsilon)
        grad[1] = (-2.0 * np.sqrt(2)) / (x[1]**2 + epsilon)
        grad[2] = (-2.0 * np.sqrt(2)) / (x[2]**2 + epsilon)
        grad[3] = 1.0
        return grad

    def hess_f2(self, x_vars):
        """
        Hessian of f2(x).
        H_f2 = diag(4/x1^3, 4*sqrt(2)/x2^3, 4*sqrt(2)/x3^3, 0)
        """
        x = np.asarray(x_vars)
        epsilon = 1e-9
        hess = np.zeros((self.n, self.n))
        hess[0,0] = 4.0 / (x[0]**3 + epsilon)
        hess[1,1] = (4.0 * np.sqrt(2)) / (x[1]**3 + epsilon)
        hess[2,2] = (4.0 * np.sqrt(2)) / (x[2]**3 + epsilon)
        # hess[3,3] is 0, already initialized
        return hess

    def calculate_optimal_pareto_front(self):
        """
        Calculates the true Pareto front for the SD problem.
        The Pareto front is given by:
        f1 = 6*t + 1
        f2 = 6/t + 1
        for t in [1, 3/sqrt(2)]
        """
        if self.num_pf_samples <= 0:
            return np.array([]) # Return empty if no sampling requested
            
        # Parameter t for the Pareto front
        t_min = 1.0
        t_max = 3.0 / np.sqrt(2)
        
        if self.num_pf_samples == 1: # Handle edge case of single sample
            t_values = np.array([t_min]) # Or average, or t_min
        else:
            t_values = np.linspace(t_min, t_max, self.num_pf_samples)
        
        f1_pf = 6.0 * t_values + 1.0
        f2_pf = 6.0 / t_values + 1.0
        
        # Stacking them column-wise: each row is a point (f1, f2)
        pareto_front_points = np.stack((f1_pf, f2_pf), axis=-1)
        return pareto_front_points

    def evaluate_f(self, x):
        """
        Evaluates all objective functions at point x.
        Returns a numpy array: [f1(x), f2(x)]
        """
        return np.array([self.f1(x), self.f2(x)])

    def evaluate_gradients_f(self, x):
        """
        Evaluates the gradients of all objective functions at point x.
        Returns a numpy array: [grad_f1(x), grad_f2(x)]
        """
        return np.array([self.grad_f1(x), self.grad_f2(x)])

    def evaluate_hessians_f(self, x):
        """
        Evaluates the Hessians of all objective functions at point x.
        Returns a numpy array: [hess_f1(x), hess_f2(x)]
        """
        return np.array([self.hess_f1(x), self.hess_f2(x)])

    def evaluate_g(self, z):
        """
        Evaluates the G(z) part of the objective F(x) + G(z).
        Currently only supports g_type 'zero'.
        """
        # Assuming z has the same dimension structure if needed, or is compatible.
        # For 'zero' g_type, g_i(z_i) = 0 for all i.
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
        """
        Evaluates the combined objective F(x) + G(z).
        """
        f_values = self.evaluate_f(x)
        g_values = self.evaluate_g(z) # z would be the input for the G functions
        return f_values + g_values

    def f_list(self):
        """
        Returns a list of tuples, where each tuple contains (fi, grad_fi, hess_fi)
        for each objective function fi.
        """
        return [
            (self.f1, self.grad_f1, self.hess_f1),
            (self.f2, self.grad_f2, self.hess_f2)
        ]

