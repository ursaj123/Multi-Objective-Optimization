import numpy as np

class TOI4:
    def __init__(self, n=4, lb=-3, ub=3, g_type=('zero', {}), fact=1):
        r"""
        TOI4 Problem (User Updated Formulation)

        Objective Functions:
        f1(x) = x1^2 + x2^2 + 1
        f2(x) = 0.5 * ((x1 - x2)^2 + (x3 - x4)^2) + 1

        Variables: x = (x1, x2, x3, x4)^T
        Default Bounds: x_i in [-5, 5] for i=0,1,2,3 (using 0-based indexing for x)
        """
        self.m = 2  # Number of objectives
        if n != 4:
            raise ValueError("TOI4 problem is defined for n=4 variables.")
        self.n = n
        
        # Store bounds for Pareto front calculation (U in analysis)
        self.lb_val = lb 
        self.ub_val = ub

        self.lb_arr = np.array([lb] * self.n) 
        self.ub_arr = np.array([ub] * self.n)
        
        self.bounds = [(self.lb_arr[i], self.ub_arr[i]) for i in range(self.n)]
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

    def f1(self, x_vars):
        """
        Objective function f1(x) = x_0^2 + x_1^2 + 1
        (using 0-based indexing for x_vars)
        """
        x = np.asarray(x_vars)
        return x[0]**2 + x[1]**2 + 1.0

    def grad_f1(self, x_vars):
        """
        Gradient of f1(x).
        grad_f1 = [2*x_0, 2*x_1, 0, 0]^T
        """
        x = np.asarray(x_vars)
        grad = np.zeros(self.n)
        grad[0] = 2 * x[0]
        grad[1] = 2 * x[1]
        return grad

    def hess_f1(self, x_vars):
        """
        Hessian of f1(x).
        H_f1 = diag(2, 2, 0, 0)
        """
        # x_vars is not used as the Hessian is constant for non-zero elements.
        hess = np.zeros((self.n, self.n))
        hess[0,0] = 2.0
        hess[1,1] = 2.0
        return hess

    def f2(self, x_vars):
        """
        Objective function f2(x) = 0.5 * ((x_0 - x_1)^2 + (x_2 - x_3)^2) + 1
        (using 0-based indexing for x_vars)
        """
        x = np.asarray(x_vars)
        term1_sq_diff = (x[0] - x[1])**2
        term2_sq_diff = (x[2] - x[3])**2
        return 0.5 * (term1_sq_diff + term2_sq_diff) + 1.0

    def grad_f2(self, x_vars):
        """
        Gradient of f2(x).
        grad_f2[0] = x_0 - x_1
        grad_f2[1] = x_1 - x_0
        grad_f2[2] = x_2 - x_3
        grad_f2[3] = x_3 - x_2
        """
        x = np.asarray(x_vars)
        grad = np.zeros(self.n)
        grad[0] = x[0] - x[1]
        grad[1] = x[1] - x[0] # -(x[0] - x[1])
        grad[2] = x[2] - x[3]
        grad[3] = x[3] - x[2] # -(x[2] - x[3])
        return grad

    def hess_f2(self, x_vars):
        """
        Hessian of f2(x).
        H_f2 = [[ 1, -1,  0,  0],
                [-1,  1,  0,  0],
                [ 0,  0,  1, -1],
                [ 0,  0, -1,  1]]
        """
        # x_vars is not used as the Hessian is constant.
        hess = np.zeros((self.n, self.n))
        
        hess[0,0] =  1.0
        hess[0,1] = -1.0
        hess[1,0] = -1.0
        hess[1,1] =  1.0
        
        hess[2,2] =  1.0
        hess[2,3] = -1.0
        hess[3,2] = -1.0
        hess[3,3] =  1.0
        return hess

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
        """
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
        g_values = self.evaluate_g(z)
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

