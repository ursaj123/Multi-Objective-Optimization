import numpy as np

class ZDT2:
    r"""
    Zitzler-Deb-Thiele Problem 2 (ZDT2).

    This is a 2-objective problem typically defined with n=30 variables.
    The objectives are:
    f1(X) = X[0]
    f2(X) = g(X) * (1 - (X[0] / g(X))^2)
    
    where:
        g(X) = 1 + (9 / (n - 1)) * sum_{i=2}^{n} x_i 
               (if X is 0-indexed, sum is over X[1]...X[n-1])
        Decision variables X[i] are in [0, 1].
        The number of variables n must be >= 2.
    
    The Pareto-optimal front is non-convex and corresponds to g(X)=1.
    On this front, f1 = X[0] and f2 = 1 - X[0]^2, with X[0] in [0,1].
    """

    def __init__(self, n=30, g_type=('zero', {})):
        """
        Initialize the ZDT2 problem.

        Args:
            n (int): Number of decision variables. Must be >= 2. Default is 30.
            g_type (tuple): Specifies the type and parameters of an additive g(z) function,
                            as used in F(X,z) = f(X) + g(z) optimization frameworks.
                            Default is ('zero', {}), meaning no additive g(z).
        """
        if not isinstance(n, int) or n < 2:
            raise ValueError("Number of variables 'n' for ZDT2 must be an integer >= 2.")
        
        self.m = 2  # Number of objectives
        self.n = n  # Number of decision variables

        self.bounds = [(0.0, 1.0) for _ in range(self.n)]
        self.constraints = []  # No explicit constraints other than bounds

        self.g_type = g_type  # For the f(X) + g(z) structure

        self.true_pareto_front = self.calculate_optimal_pareto_front(num_points=100)
        
        if self.true_pareto_front is not None and self.true_pareto_front.size > 0:
            self.ref_point = np.max(self.true_pareto_front, axis=0) + 1e-4 
        else: 
            self.ref_point = np.array([1.0 + 1e-4, 1.0 + 1e-4])

        self.l1_ratios = np.array([])
        self.l1_shifts = np.array([])
        if self.g_type[0] == 'L1':
            ratios, shifts = [], []
            for i in range(self.m):
                ratios.append(1.0 / ((i + 1) * self.m))
                shifts.append(float(i))
            self.l1_ratios = np.array(ratios)
            self.l1_shifts = np.array(shifts)
        
    def _g_zdt2(self, X_arr):
        """ 
        Calculates the ZDT-specific g(X) function. g(X) >= 1.
        X_arr is 0-indexed. The sum is over elements corresponding to x_2 to x_n.
        """
        # self.n >= 2 is guaranteed by __init__
        sum_val = np.sum(X_arr[1:]) # Sums X[1] through X[n-1]
        return 1.0 + (9.0 / (self.n - 1)) * sum_val

    def f1(self, X):
        """ Calculates the first objective: f1(X) = X[0]. """
        if not isinstance(X, np.ndarray): X = np.array(X, dtype=float)
        return X[0]

    def f2(self, X):
        """ Calculates the second objective: f2(X) = g(X) * (1 - (X[0]/g(X))^2). """
        if not isinstance(X, np.ndarray): X = np.array(X, dtype=float)
        x1 = X[0]
        g_val = self._g_zdt2(X) # g_val >= 1.0 for ZDT2

        # Since x1 >= 0 and g_val >= 1, ratio = x1/g_val is >= 0.
        ratio = x1 / g_val 
        return g_val * (1.0 - ratio**2)

    def grad_f1(self, X):
        """ Gradient of f1(X). grad_f1[0] = 1, others are 0. """
        grad = np.zeros(self.n)
        grad[0] = 1.0
        return grad

    def grad_f2(self, X):
        """ 
        Gradient of f2(X).
        f2(X) = g(X) - X[0]^2 / g(X)
        d_f2/d_X[0] = -2*X[0]/g(X)
        d_f2/d_X[k] = (dg/dX[k]) * (1 + (X[0]/g(X))^2) for k >= 1
        """
        if not isinstance(X, np.ndarray): X = np.array(X, dtype=float)
        grad = np.zeros(self.n)
        x1 = X[0]
        g_val = self._g_zdt2(X) # g_val >= 1.0

        # Component df2/dX[0]
        grad[0] = -2.0 * x1 / g_val

        # Components df2/dX[k] for k=1 to n-1 (original x_2 to x_n)
        C_dg = 9.0 / (self.n - 1) # This is dg/dX[k] for k >= 1
        
        term_factor = 1.0 + (x1 / g_val)**2

        for k in range(1, self.n): # For X[1] through X[n-1]
            grad[k] = C_dg * term_factor
            
        return grad

    def hess_f1(self, X):
        """ Hessian of f1(X). It's a zero matrix. """
        return np.zeros((self.n, self.n))

    def hess_f2(self, X):
        """ 
        Hessian of f2(X).
        d2f2/dX[0]^2 = -2/g
        d2f2/dX[k]dX[0] = 2 * C_dg * X[0] / g^2  (for k >= 1)
        d2f2/dX[j]dX[k] = -2 * C_dg^2 * X[0]^2 / g^3 (for j,k >= 1)
        """
        if not isinstance(X, np.ndarray): X = np.array(X, dtype=float)
        hess = np.zeros((self.n, self.n))
        x1 = X[0]
        g_val = self._g_zdt2(X) # g_val >= 1.0

        # H[0,0] = d^2 f2 / dX[0]^2
        hess[0, 0] = -2.0 / g_val

        # Constant C_dg = dg/dX[k] for k >= 1
        C_dg = 9.0 / (self.n - 1)
        
        # H[k,0] = H[0,k] = d^2 f2 / dX[k]dX[0] for k >= 1
        val_H_k0 = 2.0 * C_dg * x1 / (g_val**2)
        
        for k in range(1, self.n): # For variables X[1] through X[n-1]
            hess[k, 0] = val_H_k0
            hess[0, k] = val_H_k0

        # H[k,j] = d^2 f2 / dX[j]dX[k] for j,k >= 1
        val_H_kj = -2.0 * (C_dg**2) * (x1**2) / (g_val**3)

        for k in range(1, self.n):
            for j in range(1, self.n):
                hess[k, j] = val_H_kj
                
        return hess

    def calculate_optimal_pareto_front(self, num_points=100):
        """
        Calculates points on the true Pareto front for ZDT2.
        On the front: f1 = x1 (with x1 in [0,1]), f2 = 1 - x1^2.
        This occurs when X[k] = 0 for k=1,...,n-1 (i.e., x_2 to x_n are zero), making g(X)=1.
        """
        if num_points <= 0: 
            return np.array([])
        
        pareto_front = np.zeros((num_points, self.m))
        f1_values = np.linspace(0.0, 1.0, num_points)
        
        for i, f1_val in enumerate(f1_values):
            pareto_front[i, 0] = f1_val
            pareto_front[i, 1] = 1.0 - f1_val**2
        return pareto_front

    def evaluate_f(self, X):
        """Evaluates all objective functions for vector X."""
        return np.array([self.f1(X), self.f2(X)])

    def evaluate_gradients_f(self, X):
        """Evaluates gradients of all objective functions for vector X."""
        return np.array([self.grad_f1(X), self.grad_f2(X)])

    def evaluate_hessians_f(self, X):
        """Evaluates Hessians of all objective functions for vector X."""
        return np.array([self.hess_f1(X), self.hess_f2(X)])

    def evaluate_g(self, z_vec): # This is the additive g(z), not ZDT2's internal g(X)
        """
        Evaluates the problem's additive g(z) functions (e.g. for L1 penalty).
        'z_vec' is expected to be an n-dimensional vector like X.
        """
        if self.g_type[0] == 'zero':
            return np.zeros(self.m)
        elif self.g_type[0] == 'L1':
            if not self.l1_ratios.size and not self.l1_shifts.size:
                temp_ratios, temp_shifts = [], []
                for i in range(self.m):
                    temp_ratios.append(1.0 / ((i + 1) * self.m))
                    temp_shifts.append(float(i))
                self.l1_ratios = np.array(temp_ratios)
                self.l1_shifts = np.array(temp_shifts)
            
            res = np.zeros(self.m)
            if not isinstance(z_vec, np.ndarray):
                z_vec = np.array(z_vec, dtype=float)
            
            for i in range(self.m):
                res[i] = np.linalg.norm((z_vec - self.l1_shifts[i]) * self.l1_ratios[i], ord=1)
            return res
        return np.zeros(self.m) 

    def evaluate(self, X, z_vec):
        """Evaluates F(X, z_vec) = f(X) + g(z_vec)."""
        f_values = self.evaluate_f(X)
        g_values = self.evaluate_g(z_vec) 
        return f_values + g_values

    def f_list(self):
        """Returns a list of (objective, gradient, Hessian) function tuples."""
        return [
            (self.f1, self.grad_f1, self.hess_f1),
            (self.f2, self.grad_f2, self.hess_f2)
        ]