import numpy as np

class FF:
    r"""
    Fonseca-Fleming (FF) Problem.

    Minimize:
    f_1(x) = 1 - exp( -sum_{i=1}^{n} (x_i - 1/sqrt(n))^2 )
    f_2(x) = 1 - exp( -sum_{i=1}^{n} (x_i + 1/sqrt(n))^2 )

    Subject to:
    -4 <= x_i <= 4, for i = 1, ..., n
    """

    def __init__(self, n=2, lb=-4, ub=4, g_type=('zero', {})):
        """
        Initialize the FF problem.

        Arguments:
        n (int): Number of variables. Default is 2.
        lb (float): Lower bound for variables. Default is -4.
        ub (float): Upper bound for variables. Default is 4.
        g_type (tuple): Type of g function ('zero', 'L1', 'indicator', 'max')
                        and its parameters. Default is ('zero', {}).
        """
        self.m = 2  # Number of objectives
        self.n = n
        if self.n <= 0:
            raise ValueError("Number of variables 'n' must be positive.")
        self.inv_sqrt_n = 1.0 / np.sqrt(self.n)

        self.lb = lb
        self.ub = ub
        self.bounds = [(self.lb, self.ub) for _ in range(self.n)]
        self.constraints = []  # No explicit constraints other than bounds

        self.g_type = g_type

        # For Pareto front calculation (e.g., 100 points)
        self.true_pareto_front = self.calculate_optimal_pareto_front(num_points=100)
        
        if self.true_pareto_front is not None and self.true_pareto_front.size > 0:
            # Dynamically set ref_point based on the calculated Pareto front
            self.ref_point = np.max(self.true_pareto_front, axis=0) + 1e-4
        else:
            # Fallback reference point if Pareto front isn't generated or is empty
            # Max value of f_i is 1 - exp(-4) ~ 0.9817. A common ref point is [1,1] or slightly above max.
            self.ref_point = np.array([1.0 - np.exp(-4.0) + 1e-4, 1.0 - np.exp(-4.0) + 1e-4])


        self.l1_ratios, self.l1_shifts = [], []
        if self.g_type[0] == 'L1':
            # Default L1 parameter initialization (matches JOS1 style)
            # params = self.g_type[1] if len(self.g_type) > 1 else {} # For future flexibility
            for i in range(self.m):
                self.l1_ratios.append(1.0 / ((i + 1) * self.m))
                self.l1_shifts.append(float(i))
            self.l1_ratios = np.array(self.l1_ratios)
            self.l1_shifts = np.array(self.l1_shifts)

    def _S1(self, x):
        """Helper function for sum of squares term in f1."""
        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype=float)
        return np.sum((x - self.inv_sqrt_n)**2)

    def _S2(self, x):
        """Helper function for sum of squares term in f2."""
        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype=float)
        return np.sum((x + self.inv_sqrt_n)**2)

    def f1(self, x):
        """Calculates the first objective function value."""
        return 1.0 - np.exp(-self._S1(x))

    def f2(self, x):
        """Calculates the second objective function value."""
        return 1.0 - np.exp(-self._S2(x))

    def grad_f1(self, x):
        """Calculates the gradient of the first objective function."""
        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype=float)
        s1_val = self._S1(x)
        exp_s1 = np.exp(-s1_val)
        grad = 2.0 * exp_s1 * (x - self.inv_sqrt_n)
        return grad

    def grad_f2(self, x):
        """Calculates the gradient of the second objective function."""
        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype=float)
        s2_val = self._S2(x)
        exp_s2 = np.exp(-s2_val)
        grad = 2.0 * exp_s2 * (x + self.inv_sqrt_n)
        return grad

    def hess_f1(self, x):
        """Calculates the Hessian of the first objective function."""
        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype=float)
        s1_val = self._S1(x)
        exp_s1 = np.exp(-s1_val)
        
        # u_i = x_i - 1/sqrt(n)
        u_vec = x - self.inv_sqrt_n
        
        # H_kj = 2 * exp_s1 * (delta_kj - 2 * u_k * u_j)
        # H = 2 * exp_s1 * (I - 2 * u * u^T)
        outer_product_term = -2.0 * np.outer(u_vec, u_vec)
        hessian = 2.0 * exp_s1 * (np.eye(self.n) + outer_product_term)
        return hessian

    def hess_f2(self, x):
        """Calculates the Hessian of the second objective function."""
        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype=float)
        s2_val = self._S2(x)
        exp_s2 = np.exp(-s2_val)
        
        # v_i = x_i + 1/sqrt(n)
        v_vec = x + self.inv_sqrt_n
        
        # H_kj = 2 * exp_s2 * (delta_kj - 2 * v_k * v_j)
        # H = 2 * exp_s2 * (I - 2 * v * v^T)
        outer_product_term = -2.0 * np.outer(v_vec, v_vec)
        hessian = 2.0 * exp_s2 * (np.eye(self.n) + outer_product_term)
        return hessian

    def calculate_optimal_pareto_front(self, num_points=100):
        """
        Calculates points on the true Pareto front.
        For FF, optimal solutions are x_i = alpha for all i,
        where alpha is in [-1/sqrt(n), 1/sqrt(n)].
        """
        if self.n <= 0: return np.array([])
        if num_points <=0: return np.array([])
            
        pareto_front = np.zeros((num_points, self.m))
        alphas = np.linspace(-self.inv_sqrt_n, self.inv_sqrt_n, num_points)
        
        for i, alpha in enumerate(alphas):
            x_opt = np.full(self.n, alpha) # x_i = alpha for all i
            pareto_front[i, 0] = self.f1(x_opt)
            pareto_front[i, 1] = self.f2(x_opt)
        return pareto_front

    def evaluate_f(self, x):
        """Evaluates all objective functions."""
        return np.array([self.f1(x), self.f2(x)])

    def evaluate_gradients_f(self, x):
        """Evaluates gradients of all objective functions."""
        return np.array([self.grad_f1(x), self.grad_f2(x)])

    def evaluate_hessians_f(self, x):
        """Evaluates Hessians of all objective functions."""
        return np.array([self.hess_f1(x), self.hess_f2(x)])

    def evaluate_g(self, z):
        """
        Evaluates the g(z) functions.
        This implementation is generic based on g_type.
        """
        if self.g_type[0] == 'zero':
            return np.zeros(self.m)
        elif self.g_type[0] == 'L1':
            # Ensure L1 parameters are initialized
            if not self.l1_ratios.size and not self.l1_shifts.size: # Check if arrays are empty
                 # This case should ideally be handled by __init__
                 # Or if g_type can be changed post-init, re-initialize here.
                 # For now, assume __init__ has set them if g_type[0] was 'L1'.
                 # If they are still empty, it implies g_type wasn't 'L1' at init
                 # or self.m was 0, which is unlikely given self.m=2.
                 # Fallback to re-calculating defaults if absolutely necessary (e.g. dynamic change of g_type)
                temp_ratios, temp_shifts = [], []
                for i in range(self.m):
                    temp_ratios.append(1.0 / ((i + 1) * self.m))
                    temp_shifts.append(float(i))
                self.l1_ratios = np.array(temp_ratios)
                self.l1_shifts = np.array(temp_shifts)

            res = np.zeros(self.m)
            if not isinstance(z, np.ndarray):
                z = np.array(z, dtype=float)
            for i in range(self.m):
                # Assumes z is an n-dimensional vector, and l1_shifts[i] is a scalar shift.
                # l1_ratios[i] is a scalar weight.
                res[i] = np.linalg.norm((z - self.l1_shifts[i]) * self.l1_ratios[i], ord=1)
            return res
        elif self.g_type[0] == 'indicator':
            # Placeholder: Actual implementation would depend on params in self.g_type[1]
            # print(f"Warning: '{self.g_type[0]}' g_type not fully implemented for {self.__class__.__name__}. Returning zeros.")
            return np.zeros(self.m)
        elif self.g_type[0] == 'max':
            # Placeholder: Actual implementation would depend on params in self.g_type[1]
            # print(f"Warning: '{self.g_type[0]}' g_type not fully implemented for {self.__class__.__name__}. Returning zeros.")
            return np.zeros(self.m)
        else:
            # print(f"Warning: Unknown g_type '{self.g_type[0]}' for {self.__class__.__name__}. Returning zeros.")
            return np.zeros(self.m)

    def evaluate(self, x, z):
        """Evaluates F(x,z) = f(x) + g(z)."""
        f_values = self.evaluate_f(x)
        g_values = self.evaluate_g(z)
        return f_values + g_values

    def f_list(self):
        """Returns a list of (objective, gradient, Hessian) function tuples."""
        return [
            (self.f1, self.grad_f1, self.hess_f1),
            (self.f2, self.grad_f2, self.hess_f2)
        ]