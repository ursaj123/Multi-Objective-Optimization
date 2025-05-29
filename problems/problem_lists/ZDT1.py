import numpy as np

class ZDT1:
    r"""
    Zitzler-Deb-Thiele Problem 1 (ZDT1).

    This is a 2-objective problem typically defined with n=30 variables.
    The objectives are:
    f1(X) = X[0]
    f2(X) = g(X) * (1 - sqrt(X[0] / g(X)))
    
    where:
        g(X) = 1 + (9 / (n - 1)) * sum_{k=1}^{n-1} X[k]  (for 0-indexed X, sum is X[1]...X[n-1])
        Decision variables X[i] are in [0, 1].
        The number of variables n must be >= 2.
    
    The Pareto-optimal front corresponds to g(X)=1 (i.e., X[k]=0 for k=1,...,n-1).
    On this front, f1 = X[0] and f2 = 1 - sqrt(X[0]), with X[0] in [0,1].
    """

    def __init__(self, n=30, g_type=('zero', {})):
        """
        Initialize the ZDT1 problem.

        Args:
            n (int): Number of decision variables. Must be >= 2. Default is 30.
            g_type (tuple): Specifies the type and parameters of an additive g(z) function,
                            as used in F(X,z) = f(X) + g(z) optimization frameworks.
                            Default is ('zero', {}), meaning no additive g(z).
        """
        if not isinstance(n, int) or n < 2:
            raise ValueError("Number of variables 'n' for ZDT1 must be an integer >= 2.")
        
        self.m = 2  # Number of objectives
        self.n = n  # Number of decision variables

        self.bounds = [(0.0, 1.0) for _ in range(self.n)]
        self.constraints = []  # No explicit constraints other than bounds

        self.g_type = g_type  # For the f(X) + g(z) structure

        # Calculate true Pareto front (e.g., 100 points)
        self.true_pareto_front = self.calculate_optimal_pareto_front(num_points=100)
        
        # Set reference point for performance metrics (e.g., hypervolume)
        if self.true_pareto_front is not None and self.true_pareto_front.size > 0:
            # Dynamically set based on Pareto front
            self.ref_point = np.max(self.true_pareto_front, axis=0) + 1e-4 
        else: 
            # Fallback if Pareto front is empty (e.g. num_points=0)
            # For ZDT1, objectives are in [0,1] on PF.
            self.ref_point = np.array([1.0 + 1e-4, 1.0 + 1e-4])

        # Initialize parameters for L1-type additive g(z) function
        self.l1_ratios = np.array([])
        self.l1_shifts = np.array([])
        if self.g_type[0] == 'L1':
            ratios, shifts = [], []
            for i in range(self.m): # self.m is 2
                ratios.append(1.0 / ((i + 1) * self.m))
                shifts.append(float(i))
            self.l1_ratios = np.array(ratios)
            self.l1_shifts = np.array(shifts)
        
        self._eps = 1e-12  # Small epsilon for numerical stability with X[0]

    def _g_zdt1(self, X_arr):
        """ 
        Calculates the ZDT1-specific g(X) function.
        g(X) = 1 + (9 / (n - 1)) * sum X[k] for k from 1 to n-1 (0-indexed X_arr)
        Since X_arr[k] >= 0, g(X) >= 1.0.
        """
        # self.n >= 2 is guaranteed by __init__
        sum_val = np.sum(X_arr[1:]) # Sums X[1] up to X[n-1]
        return 1.0 + (9.0 / (self.n - 1)) * sum_val

    def f1(self, X):
        """ Calculates the first objective: f1(X) = X[0]. """
        if not isinstance(X, np.ndarray): X = np.array(X, dtype=float)
        return X[0]

    def f2(self, X):
        """ Calculates the second objective: f2(X) = g(X) * (1 - sqrt(X[0]/g(X))). """
        if not isinstance(X, np.ndarray): X = np.array(X, dtype=float)
        x1 = X[0]
        g_val = self._g_zdt1(X) # g_val >= 1.0 for ZDT1

        ratio = x1 / g_val # Since x1 >= 0 and g_val >= 1, ratio >= 0.
        # Ensure ratio is not negative due to minor precision artifacts if x1 is tiny negative
        if ratio < 0.0: ratio = 0.0 

        return g_val * (1.0 - np.sqrt(ratio))

    def grad_f1(self, X):
        """ Gradient of f1(X). grad_f1[0] = 1, others are 0. """
        grad = np.zeros(self.n)
        grad[0] = 1.0
        return grad

    def grad_f2(self, X):
        """ Gradient of f2(X). """
        if not isinstance(X, np.ndarray): X = np.array(X, dtype=float)
        grad = np.zeros(self.n)
        x1 = X[0]
        g_val = self._g_zdt1(X) # g_val >= 1.0

        # Component df2/dX[0]
        if x1 < self._eps:
            grad[0] = -np.inf  # Derivative tends to -infinity as x1 -> 0+
        else: # x1 >= eps, g_val >= 1
            grad[0] = -0.5 * np.sqrt(g_val / x1)
        
        # Components df2/dX[j] for j=1 to n-1 (original x_2 to x_n)
        dg_dxj_factor = 9.0 / (self.n - 1) # This is dg/dX[j] for j > 0
        
        term_in_parenthesis = 0.0
        if x1 < self._eps: # If X[0] is zero
            term_in_parenthesis = 1.0 # The term becomes (1 - 0)
        else: # X[0] >= eps, g_val >= 1.0
            # Term is 1 - (sqrt(X[0]) / (2 * sqrt(g(X))))
            term_in_parenthesis = 1.0 - 0.5 * np.sqrt(x1 / g_val)

        for j in range(1, self.n): # For X[1] through X[n-1]
            grad[j] = dg_dxj_factor * term_in_parenthesis
            
        return grad

    def hess_f1(self, X):
        """ Hessian of f1(X). It's a zero matrix. """
        return np.zeros((self.n, self.n))

    def hess_f2(self, X):
        """ Hessian of f2(X). """
        if not isinstance(X, np.ndarray): X = np.array(X, dtype=float)
        hess = np.zeros((self.n, self.n))
        x1 = X[0]
        g_val = self._g_zdt1(X) # g_val >= 1.0

        # H[0,0] = d^2 f2 / dX[0]^2
        if x1 < self._eps:
            hess[0, 0] = np.inf # Tends to +infinity
        else: # x1 >= eps, g_val >= 1.0
            hess[0, 0] = 0.25 * np.sqrt(g_val) * (x1**(-1.5))

        # Constant C_deriv = dg/dX[j] for j > 0 (i.e. for variables X[1]...X[n-1])
        C_deriv = 9.0 / (self.n - 1)
        
        # H[j,0] = H[0,j] = d^2 f2 / dX[j]dX[0] for j > 0
        val_H_j0 = 0.0
        if x1 < self._eps:
            val_H_j0 = -np.inf # Tends to -infinity 
        else: # x1 >= eps, g_val >= 1.0
            val_H_j0 = -0.25 * C_deriv * (x1**(-0.5)) * (g_val**(-0.5))
        
        for j in range(1, self.n): # For variables X[1] through X[n-1]
            hess[j, 0] = val_H_j0
            hess[0, j] = val_H_j0

        # H[j,k] = d^2 f2 / dX[k]dX[j] for j,k > 0
        val_H_jk = 0.0
        if x1 < self._eps: # If x1 is zero, this term is zero
            val_H_jk = 0.0
        else: # x1 >= eps, g_val >= 1.0
            val_H_jk = 0.25 * (C_deriv**2) * np.sqrt(x1) * (g_val**(-1.5))

        for j in range(1, self.n):
            for k in range(1, self.n):
                hess[j, k] = val_H_jk
                
        return hess

    def calculate_optimal_pareto_front(self, num_points=100):
        """
        Calculates points on the true Pareto front for ZDT1.
        On the front: f1 = x1 (with x1 in [0,1]), f2 = 1 - sqrt(f1).
        This occurs when X[k] = 0 for k=1,...,n-1 (i.e., x_2 to x_n are zero), making g(X)=1.
        """
        if num_points <= 0: 
            return np.array([])
        
        pareto_front = np.zeros((num_points, self.m))
        f1_values = np.linspace(0.0, 1.0, num_points)
        
        for i, f1_val in enumerate(f1_values):
            pareto_front[i, 0] = f1_val
            pareto_front[i, 1] = 1.0 - np.sqrt(f1_val)
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

    def evaluate_g(self, z_vec): # This is the additive g(z), not ZDT1's internal g(X)
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
        # Fallback for other unimplemented g_types
        # print(f"Warning: g_type '{self.g_type[0]}' not fully implemented for {self.__class__.__name__}. Returning zeros.")
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