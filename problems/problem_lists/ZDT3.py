import numpy as np
from scipy.optimize import minimize

class ZDT3:
    def __init__(self, n=30, lb=0, ub=1, g_type=('zero', {})):
        """
        ZDT3 Problem

        f_1(x_1) = x_1
        g(x) = 1 + (9/(n-1)) * sum_{i=2}^{n} x_i
        h(f_1, g) = 1 - sqrt(f_1 / g) - (f_1 / g) * sin(10 * pi * f_1)
        f_2(x) = g(x) * h(f_1, g)

        Pareto front is formed when g(x) = 1

        Arguments:
        m: Number of objectives (fixed to 2 for ZDT3)
        n: Number of variables
        ub: Upper bound for variables
        g_type: Type of g function
        It can be one of ('zero', 'L1', 'indicator', 'max')
        Its parameters must be defined in the dictionary

        Defined Vars:
        bounds: Bounds for variables [(low, high), ...]
        constraints: List of constraints for the problem
        """
        self.m = 2
        self.n = n
        self.lb = lb
        self.ub = ub
        self.bounds = tuple([(lb, ub) for _ in range(n)])
        self.constraints = []
        self.g_type = g_type
        self.true_pareto_front = self.calculate_optimal_pareto_front()
        self.ref_point = np.max(self.true_pareto_front, axis=0) + 1e-4

    def g(self, x):
        """Calculates the g(x) function for ZDT3."""
        return 1 + (9 / (self.n - 1)) * np.sum(x[1:])

    def h(self, f1, g):
        """Calculates the h(f1, g) function for ZDT3."""
        return 1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1)

    def f1(self, x):
        """Calculates the first objective function f1(x) for ZDT3."""
        return x[0]

    def grad_f1(self, x):
        """Gradient of f1."""
        grad = np.zeros(self.n)
        grad[0] = 1
        return grad

    def f2(self, x):
        """Calculates the second objective function f2(x) for ZDT3."""
        g_val = self.g(x)
        h_val = self.h(x[0], g_val)
        return g_val * h_val

    def grad_f2(self, x):
        """Gradient of f2."""
        g_val = self.g(x)
        h_val = self.h(x[0], g_val)
        grad_g = np.zeros(self.n)
        grad_g[1:] = 9 / (self.n - 1)

        grad_h = np.zeros(self.n)
        df1 = x[0]
        grad_h[0] = -0.5 * (g_val * df1)**(-0.5) + 0.5 * df1**0.5 * g_val**(-1.5) * grad_g[0] - (1/g_val**2) * np.sin(10 * np.pi * df1) - (df1/g_val) * 10 * np.pi * np.cos(10 * np.pi * df1) + (df1**2/g_val**3) * np.sin(10 * np.pi * df1) * grad_g[0]
        grad_h[1:] =  0.5 * df1**0.5 * g_val**(-1.5) * grad_g[1:] + (df1**2/g_val**3) * np.sin(10 * np.pi * df1) * grad_g[1:]

        grad_f2 = h_val * grad_g + g_val * grad_h
        return grad_f2

    def hess_f1(self, x):
        """Hessian of f1."""
        return np.zeros((self.n, self.n))

    def hess_f2(self, x):
        """Hessian of f2."""
        g_val = self.g(x)
        f1 = x[0]
        n = self.n
        grad_g = np.zeros(n)
        grad_g[1:] = 9 / (n - 1)

        h_val = self.h(f1, g_val)
        dh_df1 = -0.5 * (g_val * f1)**(-0.5) - (f1 / g_val) * np.sin(10 * np.pi * f1)
        dh_dg = 0.5 * f1**0.5 * g_val**(-1.5) + (f1**2/g_val**3) * np.sin(10 * np.pi * f1)

        d2h_df12 = 1/(4*f1*g_val)**0.5 - (2/g_val) * 10 * np.pi * np.cos(10 * np.pi * f1) - (f1/g_val) * (10 * np.pi)**2 * np.sin(10 * np.pi * f1)
        d2h_df1dg = -1/(4*g_val**2 * (f1/g_val)**0.5) + (2*f1)/g_val**3 * np.sin(10 * np.pi * f1) - (f1**2)/g_val**4 * np.sin(10 * np.pi * f1) + (f1/g_val**2) * 10 * np.pi * np.cos(10*np.pi*f1)
        d2h_dg2 = (3 * f1)/(4 * g_val**(5/2) * np.sqrt(f1)) - (3*f1**2)/(g_val**4) * np.sin(10 * np.pi * f1)
        
        hess_g = np.zeros((n,n))
        hess_h = np.zeros((n,n))
        hess_h[0,0] = d2h_df12
        hess_h[0,1:] = d2h_df1dg
        hess_h[1:,0] = d2h_df1dg
        hess_h[1:,1:] = np.outer(grad_g[1:],grad_g[1:]) * d2h_dg2

        hess_f2 = h_val * hess_g + np.outer(grad_g, self.grad_f1(x)) * dh_dg + np.outer(self.grad_f1(x), grad_g) * dh_dg + g_val * hess_h
        return hess_f2

    def evaluate_f(self, x):
        """Evaluates both objective functions."""
        return np.array([self.f1(x), self.f2(x)])

    def evaluate_gradients_f(self, x):
        """Evaluates the gradients of both objective functions."""
        return np.array([self.grad_f1(x), self.grad_f2(x)])

    def evaluate_hessians_f(self, x):
        """Evaluates the hessians of both objective functions."""
        return np.array([self.hess_f1(x), self.hess_f2(x)])

    def evaluate_g(self, z):
        """Evaluates the constraint function g."""
        if self.g_type[0] == 'zero':
            return np.zeros(self.m)
        elif self.g_type[0] == 'L1':
            pass
        elif self.g_type[0] == 'indicator':
            pass
        else:
            pass

    def evaluate(self, x, z):
        """Evaluates the objective and constraint functions."""
        f_values = self.evaluate_f(x)
        g_values = self.evaluate_g(z)
        return f_values + g_values

    def f_list(self):
        """Returns a list of objective functions and their gradients/hessians."""
        return [
            (self.f1, self.grad_f1, self.hess_f1),
            (self.f2, self.grad_f2, self.hess_f2)
        ]

    def calculate_optimal_pareto_front(self):
        """Calculates the optimal Pareto front for ZDT3 (g(x) = 1)."""
        x1_values = np.linspace(0.1, 1, 100)  # Avoid x1 = 0 due to the sine term
        pareto_front = np.zeros((100, 2))
        pareto_front[:, 0] = x1_values
        pareto_front[:, 1] = 1 - np.sqrt(x1_values) - x1_values * np.sin(10 * np.pi * x1_values)
        return pareto_front
