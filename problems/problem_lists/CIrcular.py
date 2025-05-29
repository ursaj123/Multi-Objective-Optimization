import numpy as np

class CIRCULAR:
    def __init__(self, n=10, cond_num=1e2, lb = -1000, ub = 1000, g_type=('zero', {})):
        self.m = 1
        self.n = n
        self.bounds = None
        self.g_type = g_type
        self.cond_num = cond_num
        eigenvalues = np.logspace(0, np.log10(cond_num), n)
    
        # Random orthogonal matrix via QR decomposition of a random matrix
        A = np.random.randn(n ,n)
        Q, _ = np.linalg.qr(A)
        
        # Construct Q = V diag(eigenvalues) V^T (SPD)
        self.Q = Q @ np.diag(eigenvalues) @ Q.T
        self.b = np.random.randn(n)
        self.c = 0.5
        self.orig_soln = np.linalg.solve(self.Q, self.b)
        print(f"Original soln: {self.orig_soln}")

    def f1(self, x):
        return 0.5 * np.dot(x, self.Q @ x) - np.dot(self.b, x) + self.c
        
    def grad_f1(self, x):
        return self.Q @ x - self.b

    def hess_f1(self, x):
        return self.Q
    
    def evaluate_f(self, x):
        return np.array([self.f1(x)])

    def evaluate_gradients_f(self, x):
        return np.array([self.grad_f1(x)])

    def evaluate_hessians_f(self, x):
        return np.array([self.hess_f1(x)])

    def evaluate_g(self, z):
        if self.g_type[0] == 'zero':
            return np.zeros(self.m)
        elif self.g_type[0] == 'L1':
            pass
        elif self.g_type[0] == 'indicator':
            pass
        else:
            pass

    def evaluate(self, x, z):
        f_values = self.evaluate_functions(x)
        g_values = self.evaluate_g(z)
        return f_values + g_values    

    def f_list(self):
        return [
            (self.f1, self.grad_f1, self.hess_f1)
        ]
