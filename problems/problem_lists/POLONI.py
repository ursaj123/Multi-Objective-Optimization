import numpy as np

class POLONI:
    r"""
    Poloni's Two-Objective Problem.

    This problem is characterized by two variables x and y, and two objective functions.
    The Pareto front is known to be disconnected, consisting of four segments.

    Minimize:
    f1(x, y) = [ 1 + (A1 - B1(x,y))^2 + (A2 - B2(x,y))^2 ]
    f2(x, y) = (x+3)^2 + (y+1)^2

    where:
    A1 = 0.5 sin(1) - 2 cos(1) + sin(2) - 1.5 cos(2)
    A2 = 1.5 sin(1) - cos(1) + 2 sin(2) - 0.5 cos(2)
    B1(x,y) = 0.5 sin(x) - 2 cos(x) + sin(y) - 1.5 cos(y)
    B2(x,y) = 1.5 sin(x) - cos(x) + 2 sin(y) - 0.5 cos(y)

    Subject to:
    -pi <= x, y <= pi
    """

    def __init__(self, n=2, g_type=('zero', {})):
        """
        Initialize the POLONI problem.

        Arguments:
        g_type (tuple): Type of g function ('zero', 'L1', 'indicator', 'max')
                        and its parameters. Default is ('zero', {}).
        """
        self.m = 2  # Number of objectives
        self.n = 2  # Number of variables (x, y)
        self.var_labels = ['x', 'y'] # For clarity when dealing with variables

        # Calculate A1, A2 constants
        self.A1 = 0.5 * np.sin(1.0) - 2.0 * np.cos(1.0) + np.sin(2.0) - 1.5 * np.cos(2.0)
        self.A2 = 1.5 * np.sin(1.0) - np.cos(1.0) + 2.0 * np.sin(2.0) - 0.5 * np.cos(2.0)

        self.bounds = [(-np.pi, np.pi), (-np.pi, np.pi)]
        self.constraints = []  # No explicit constraints other than bounds

        self.g_type = g_type

        # Pareto front for Poloni is complex and typically loaded from data.
        self.true_pareto_front = self.calculate_optimal_pareto_front() # Returns empty
        
        # Reference point based on literature for Poloni problem.
        # Approximate maximums: f1 ~70-75, f2 ~55-60.
        self.ref_point = np.array([75.0, 60.0]) 

        self.l1_ratios = np.array([]) # Initialize as empty numpy array
        self.l1_shifts = np.array([]) # Initialize as empty numpy array
        if self.g_type[0] == 'L1':
            ratios, shifts = [], []
            for i in range(self.m):
                ratios.append(1.0 / ((i + 1) * self.m))
                shifts.append(float(i))
            self.l1_ratios = np.array(ratios)
            self.l1_shifts = np.array(shifts)

    # Helper methods for B1, B2 values
    def _B1_val(self, X_arr):
        x_var, y_var = X_arr[0], X_arr[1]
        return (0.5 * np.sin(x_var) - 2.0 * np.cos(x_var) +
                np.sin(y_var) - 1.5 * np.cos(y_var))

    def _B2_val(self, X_arr):
        x_var, y_var = X_arr[0], X_arr[1]
        return (1.5 * np.sin(x_var) - np.cos(x_var) +
                2.0 * np.sin(y_var) - 0.5 * np.cos(y_var))

    # Helper methods for first partial derivatives of B1, B2 components
    def _dB1_dx_val(self, x_var): # Component B1_x'(x)
        return 0.5 * np.cos(x_var) + 2.0 * np.sin(x_var)
    
    def _dB1_dy_val(self, y_var): # Component B1_y'(y)
        return np.cos(y_var) + 1.5 * np.sin(y_var)

    def _dB2_dx_val(self, x_var): # Component B2_x'(x)
        return 1.5 * np.cos(x_var) + np.sin(x_var)

    def _dB2_dy_val(self, y_var): # Component B2_y'(y)
        return 2.0 * np.cos(y_var) + 0.5 * np.sin(y_var)

    # Helper methods for second partial derivatives of B1, B2 components
    def _d2B1_dx2_val(self, x_var): # Component B1_x''(x)
        return -0.5 * np.sin(x_var) + 2.0 * np.cos(x_var)

    def _d2B1_dy2_val(self, y_var): # Component B1_y''(y)
        return -np.sin(y_var) + 1.5 * np.cos(y_var)
    
    def _d2B2_dx2_val(self, x_var): # Component B2_x''(x)
        return -1.5 * np.sin(x_var) + np.cos(x_var)

    def _d2B2_dy2_val(self, y_var): # Component B2_y''(y)
        return -2.0 * np.sin(y_var) + 0.5 * np.cos(y_var)

    # Objective functions
    def f1(self, X):
        """Calculates the first objective function value."""
        if not isinstance(X, np.ndarray): X = np.array(X, dtype=float)
        term1 = self.A1 - self._B1_val(X)
        term2 = self.A2 - self._B2_val(X)
        return 1.0 + term1**2 + term2**2

    def f2(self, X):
        """Calculates the second objective function value."""
        if not isinstance(X, np.ndarray): X = np.array(X, dtype=float)
        return (X[0] + 3.0)**2 + (X[1] + 1.0)**2

    # Gradients
    def grad_f1(self, X):
        """Calculates the gradient of the first objective function."""
        if not isinstance(X, np.ndarray): X = np.array(X, dtype=float)
        x_var, y_var = X[0], X[1]

        B1 = self._B1_val(X)
        B2 = self._B2_val(X)
        
        T1 = self.A1 - B1  # A1 - B1(x,y)
        T2 = self.A2 - B2  # A2 - B2(x,y)

        # Partial derivatives of B1 w.r.t x and y (these are B1_x'(x) and B1_y'(y))
        dB1dx = self._dB1_dx_val(x_var)
        dB1dy = self._dB1_dy_val(y_var)
        # Partial derivatives of B2 w.r.t x and y (these are B2_x'(x) and B2_y'(y))
        dB2dx = self._dB2_dx_val(x_var)
        dB2dy = self._dB2_dy_val(y_var)

        # df1/dx = 2*T1*(-dB1/dx) + 2*T2*(-dB2/dx)
        grad_x = -2.0 * T1 * dB1dx - 2.0 * T2 * dB2dx
        # df1/dy = 2*T1*(-dB1/dy) + 2*T2*(-dB2/dy)
        grad_y = -2.0 * T1 * dB1dy - 2.0 * T2 * dB2dy
        
        return np.array([grad_x, grad_y])

    def grad_f2(self, X):
        """Calculates the gradient of the second objective function."""
        if not isinstance(X, np.ndarray): X = np.array(X, dtype=float)
        grad_x = 2.0 * (X[0] + 3.0)
        grad_y = 2.0 * (X[1] + 1.0)
        return np.array([grad_x, grad_y])

    # Hessians
    def hess_f1(self, X):
        """Calculates the Hessian of the first objective function."""
        if not isinstance(X, np.ndarray): X = np.array(X, dtype=float)
        x_var, y_var = X[0], X[1]

        B1 = self._B1_val(X)
        B2 = self._B2_val(X)
        
        T1 = self.A1 - B1 
        T2 = self.A2 - B2 

        dB1dx = self._dB1_dx_val(x_var)
        d2B1dx2 = self._d2B1_dx2_val(x_var)
        dB1dy = self._dB1_dy_val(y_var)
        # d2B1dy2 = self._d2B1_dy2_val(y_var) # Not needed for h11, but for h22

        dB2dx = self._dB2_dx_val(x_var)
        d2B2dx2 = self._d2B2_dx2_val(x_var)
        dB2dy = self._dB2_dy_val(y_var)
        # d2B2dy2 = self._d2B2_dy2_val(y_var) # Not needed for h11, but for h22

        # d^2 f1 / dx^2 = 2 * [ (dB1/dx)^2 - T1*(d^2B1/dx^2) + (dB2/dx)^2 - T2*(d^2B2/dx^2) ]
        h11 = 2.0 * (dB1dx**2 - T1 * d2B1dx2 + dB2dx**2 - T2 * d2B2dx2)
        
        # d^2 f1 / dy^2
        # Need d2B1dy2 and d2B2dy2 for h22
        d2B1dy2_val = self._d2B1_dy2_val(y_var)
        d2B2dy2_val = self._d2B2_dy2_val(y_var)
        h22 = 2.0 * (dB1dy**2 - T1 * d2B1dy2_val + dB2dy**2 - T2 * d2B2dy2_val)
        
        # d^2 f1 / dx dy = 2 * [ (dB1/dx)*(dB1/dy) + (dB2/dx)*(dB2/dy) ]
        # (since d^2B1/dxdy = 0 and d^2B2/dxdy = 0)
        h12 = 2.0 * (dB1dx * dB1dy + dB2dx * dB2dy)
        
        return np.array([[h11, h12], [h12, h22]])

    def hess_f2(self, X):
        """Calculates the Hessian of the second objective function."""
        # X is not used as the Hessian is constant
        return np.array([[2.0, 0.0], [0.0, 2.0]])

    def calculate_optimal_pareto_front(self, num_points=0):
        """
        The true Pareto front for Poloni's problem is disconnected and complex,
        typically loaded from pre-computed data sets. This method returns an empty array
        as a placeholder.
        """
        # `num_points` argument is included for API consistency with other problems.
        # print(f"Note: True Pareto front for {self.__class__.__name__} is complex. Returning empty array.")
        return np.array([[0.0, 1.0], [1.0, 0.0]])

    def evaluate_f(self, X):
        """Evaluates all objective functions."""
        return np.array([self.f1(X), self.f2(X)])

    def evaluate_gradients_f(self, X):
        """Evaluates gradients of all objective functions."""
        return np.array([self.grad_f1(X), self.grad_f2(X)])

    def evaluate_hessians_f(self, X):
        """Evaluates Hessians of all objective functions."""
        return np.array([self.hess_f1(X), self.hess_f2(X)])

    def evaluate_g(self, z):
        """
        Evaluates the g(z) functions.
        This implementation is generic based on g_type.
        """
        if self.g_type[0] == 'zero':
            return np.zeros(self.m)
        elif self.g_type[0] == 'L1':
            # Ensure L1 parameters are initialized (should be by __init__)
            if not self.l1_ratios.size and not self.l1_shifts.size:
                 # This is a fallback, __init__ should handle typical cases.
                temp_ratios, temp_shifts = [], []
                for i in range(self.m): # self.m is 2 for Poloni
                    temp_ratios.append(1.0 / ((i + 1) * self.m))
                    temp_shifts.append(float(i))
                self.l1_ratios = np.array(temp_ratios)
                self.l1_shifts = np.array(temp_shifts)
            
            res = np.zeros(self.m)
            if not isinstance(z, np.ndarray):
                z = np.array(z, dtype=float) # z is expected to be like X (n-dimensional)
            
            for i in range(self.m):
                # Assumes z is an n-dimensional vector (size self.n).
                # self.l1_shifts[i] is a scalar, subtracted element-wise from z.
                # self.l1_ratios[i] is a scalar multiplier.
                res[i] = np.linalg.norm((z - self.l1_shifts[i]) * self.l1_ratios[i], ord=1)
            return res
        elif self.g_type[0] == 'indicator':
            # print(f"Warning: '{self.g_type[0]}' g_type not fully implemented for {self.__class__.__name__}. Returning zeros.")
            return np.zeros(self.m) # Placeholder
        elif self.g_type[0] == 'max':
            # print(f"Warning: '{self.g_type[0]}' g_type not fully implemented for {self.__class__.__name__}. Returning zeros.")
            return np.zeros(self.m) # Placeholder
        else:
            # print(f"Warning: Unknown g_type '{self.g_type[0]}' for {self.__class__.__name__}. Returning zeros.")
            return np.zeros(self.m) # Placeholder

    def evaluate(self, X, z):
        """Evaluates F(X,z) = f(X) + g(z)."""
        f_values = self.evaluate_f(X)
        g_values = self.evaluate_g(z) # z can be different from X
        return f_values + g_values

    def f_list(self):
        """Returns a list of (objective, gradient, Hessian) function tuples."""
        return [
            (self.f1, self.grad_f1, self.hess_f1),
            (self.f2, self.grad_f2, self.hess_f2)
        ]