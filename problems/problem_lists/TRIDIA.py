import numpy as np

class TRIDIA:
    def __init__(self, n=3, lb=-3, ub=3, g_type=('zero', {}), fact=1): # num_pf_samples is trivial here
        r"""
        TRIDIA Problem

        Objective Functions (using 0-based indexing for x_vars: x0, x1, x2):
        f1(x) = (2*x0 - 1)^2
        f2(x) = 2 * (2*x0 - x1)^2
        f3(x) = 3 * (2*x1 - x2)^2

        Variables: x = (x0, x1, x2)^T
        Default Bounds: x_i in [-5, 5]
        """
        self.m = 3  # Number of objectives
        if n != 3:
            raise ValueError("TRIDIA problem is defined for n=3 variables.")
        self.n = n
        
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
        Objective function f1(x) = (2*x0 - 1)^2
        """
        x = np.asarray(x_vars)
        return (2 * x[0] - 1)**2

    def grad_f1(self, x_vars):
        """
        Gradient of f1(x).
        grad_f1 = [4*(2*x0 - 1), 0, 0]^T
                  = [8*x0 - 4, 0, 0]^T
        """
        x = np.asarray(x_vars)
        grad = np.zeros(self.n)
        grad[0] = 8 * x[0] - 4
        return grad

    def hess_f1(self, x_vars):
        """
        Hessian of f1(x).
        H_f1 = diag(8, 0, 0)
        """
        # x_vars is not used as the Hessian is constant.
        hess = np.zeros((self.n, self.n))
        hess[0,0] = 8.0
        return hess

    def f2(self, x_vars):
        """
        Objective function f2(x) = 2 * (2*x0 - x1)^2
        """
        x = np.asarray(x_vars)
        return 2 * (2 * x[0] - x[1])**2

    def grad_f2(self, x_vars):
        """
        Gradient of f2(x).
        grad_f2[0] = 8 * (2*x0 - x1) = 16*x0 - 8*x1
        grad_f2[1] = -4 * (2*x0 - x1) = 4*x1 - 8*x0
        grad_f2[2] = 0
        """
        x = np.asarray(x_vars)
        grad = np.zeros(self.n)
        term = 2 * x[0] - x[1]
        grad[0] = 8 * term
        grad[1] = -4 * term
        return grad

    def hess_f2(self, x_vars):
        """
        Hessian of f2(x).
        H_f2 = [[ 16, -8,  0],
                [ -8,  4,  0],
                [  0,  0,  0]]
        """
        # x_vars is not used as the Hessian is constant.
        hess = np.zeros((self.n, self.n))
        hess[0,0] =  16.0
        hess[0,1] = -8.0
        hess[1,0] = -8.0
        hess[1,1] =  4.0
        return hess

    def f3(self, x_vars):
        """
        Objective function f3(x) = 3 * (2*x1 - x2)^2
        """
        x = np.asarray(x_vars)
        return 3 * (2 * x[1] - x[2])**2

    def grad_f3(self, x_vars):
        """
        Gradient of f3(x).
        grad_f3[0] = 0
        grad_f3[1] = 12 * (2*x1 - x2) = 24*x1 - 12*x2
        grad_f3[2] = -6 * (2*x1 - x2) = 6*x2 - 12*x1
        """
        x = np.asarray(x_vars)
        grad = np.zeros(self.n)
        term = 2 * x[1] - x[2]
        grad[1] = 12 * term
        grad[2] = -6 * term
        return grad

    def hess_f3(self, x_vars):
        """
        Hessian of f3(x).
        H_f3 = [[  0,  0,   0],
                [  0, 24, -12],
                [  0,-12,   6]]
        """
        # x_vars is not used as the Hessian is constant.
        hess = np.zeros((self.n, self.n))
        hess[1,1] =  24.0
        hess[1,2] = -12.0
        hess[2,1] = -12.0
        hess[2,2] =  6.0
        return hess

    def evaluate_f(self, x):
        """
        Evaluates all objective functions at point x.
        Returns a numpy array: [f1(x), f2(x), f3(x)]
        """
        return np.array([self.f1(x), self.f2(x), self.f3(x)])

    def evaluate_gradients_f(self, x):
        """
        Evaluates the gradients of all objective functions at point x.
        Returns a numpy array of shape (m, n): 
        [[grad_f1(x)], [grad_f2(x)], [grad_f3(x)]]
        """
        return np.array([self.grad_f1(x), self.grad_f2(x), self.grad_f3(x)])

    def evaluate_hessians_f(self, x):
        """
        Evaluates the Hessians of all objective functions at point x.
        Returns a numpy array of shape (m, n, n):
        [[hess_f1(x)], [hess_f2(x)], [hess_f3(x)]]
        """
        return np.array([self.hess_f1(x), self.hess_f2(x), self.hess_f3(x)])

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
            (self.f2, self.grad_f2, self.hess_f2),
            (self.f3, self.grad_f3, self.hess_f3)
        ]

