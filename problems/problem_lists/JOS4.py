import numpy as np

class JOS4:
    """
    JOS4 Problem
    Formulation:
    f1(x) = x1
    g(x2, ..., xn) = 1 + (9 / (n - 1)) * Σ(i=2 to n) xi
    f2(x) = g(x2, ..., xn) * (1.0 - (f1(x)/g(x2, ..., xn))**0.25 - (f1(x)/g(x2, ..., xn))**4 )
    """
    def __init__(self, n=2, lb=0, ub=1, g_type=('zero', {})):
        """
        Initialize the JOS4 problem.

        Args:
            n (int): Number of variables.
            lb (float): Lower bound for variables.
            ub (float): Upper bound for variables.
            g_type (tuple): Type of g function and its parameters.
                           It can be one of ('zero', 'L1', 'indicator', 'max').
                           Its parameters must be defined in the dictionary.
        """
        self.m = 2  # Number of objectives
        self.n = n
        self.lb = lb
        self.ub = ub
        self.bounds = tuple([(lb, ub) for _ in range(n)])
        self.constraints = []
        self.g_type = g_type
        self.true_pareto_front = self.calculate_optimal_pareto_front()
        self.ref_point = np.max(self.true_pareto_front, axis=0) + 1e-4


        self.l1_ratios, self.l1_shifts = [], []
        if self.g_type[0]=='L1':
            for i in range(self.m):
                self.l1_ratios.append(1/((i+1)*self.m))
                self.l1_shifts.append(i)
            self.l1_ratios = np.array(self.l1_ratios)
            self.l1_shifts = np.array(self.l1_shifts)

    def calculate_optimal_pareto_front(self, num_points=100):
        """
        Calculate the optimal Pareto front.

        Returns:
            numpy.ndarray: The Pareto front as a 2D array.
        """
        x1_values = np.linspace(0, 1, num_points)
        pareto_front = np.zeros((num_points, 2))
        for i, x1 in enumerate(x1_values):
            pareto_front[i, 0] = x1
            pareto_front[i, 1] = 1 - x1**0.25 - x1**4
        return pareto_front

    def g_function(self, x):
        """
        Calculate the g function.

        Args:
            x (numpy.ndarray): Decision vector.

        Returns:
            float: The value of the g function.
        """
        if self.n <= 1:
          return 1.0
        return 1 + (9 / (self.n - 1)) * np.sum(x[1:])

    def f1(self, x):
        """
        Calculate the first objective function.

        Args:
            x (numpy.ndarray): Decision vector.

        Returns:
            float: The value of the first objective function.
        """
        return x[0]

    def grad_f1(self, x):
        """
        Calculate the gradient of the first objective function.

        Args:
            x (numpy.ndarray): Decision vector.

        Returns:
            numpy.ndarray: The gradient of the first objective function.
        """
        grad = np.zeros(self.n)
        grad[0] = 1
        return grad

    def hess_f1(self, x):
        """
        Calculate the Hessian of the first objective function.

        Args:
            x (numpy.ndarray): Decision vector.

        Returns:
            numpy.ndarray: The Hessian of the first objective function.
        """
        return np.eye(self.n) * 0

    def f2(self, x):
        """
        Calculate the second objective function.

        Args:
            x (numpy.ndarray): Decision vector.

        Returns:
            float: The value of the second objective function.
        """
        g = self.g_function(x)
        f1 = self.f1(x)
        return g * (1.0 - (f1 / g)**0.25 - (f1 / g)**4)

    def grad_f2(self, x):
        """
        Calculate the gradient of the second objective function.

        Args:
            x (numpy.ndarray): Decision vector.

        Returns:
            numpy.ndarray: The gradient of the second objective function.
        """
        g = self.g_function(x)
        f1 = self.f1(x)
        n = self.n

        # Calculate partial derivatives of g with respect to x[1:]
        dg_dxi = 9 / (n - 1) if n > 1 else 0

        # common terms to reduce repeating calculations
        f1_g = f1 / g
        pow_025 = f1_g**0.25
        pow_4 = f1_g**4

        # Derivative of f2 with respect to x1
        df2_dx1 = g * ( (-0.25 * g**(-1) * f1**(-0.75)) - (4 * g**(-1) * f1**3) )
        df2_dx = np.zeros(n)
        df2_dx[0] = df2_dx1

        # Derivative of f2 with respect to x_i, i = 2,...,n
        if n > 1:
            for i in range(1, n):
                df2_dx_i = (dg_dxi * (1 - pow_025 - pow_4) +
                            g * (0.25 * pow_025 * f1 / g**2 + 4 * pow_4 * f1 / g**2))
                df2_dx[i] = df2_dx_i
        return df2_dx

    def hess_f2(self, x):
        """
        Calculate the Hessian of the second objective function.
        Args:
            x (numpy.ndarray): Decision vector.

        Returns:
            numpy.ndarray: The Hessian of the second objective function.
        """
        g = self.g_function(x)
        f1 = self.f1(x)
        n = self.n

        # Pre-calculate some common terms
        f1_g = f1 / g
        pow_025 = f1_g**0.25
        pow_4 = f1_g**4
        dg_dxi = 9 / (n - 1) if n > 1 else 0

        # Initialize the Hessian matrix
        hess_f2 = np.zeros((n, n))

        # Calculate the second-order partial derivatives
        # d2f2/dx1^2
        hess_f2[0, 0] = g * (
            (0.25 * 1.25 * g**(-2) * f1**(-1.75)) +
            (4 * 3 * g**(-2) * f1**2)
        )

        if n > 1:
            # d2f2/dx1dx_i and d2f2/dx_idx1 for i = 2,...,n
            for i in range(1, n):
                hess_f2[0, i] = (
                    -0.25 * dg_dxi * f1**(-0.75) * g**(-1)
                    + 0.25 * 0.75 * f1**(-0.75) * g**(-2) * f1 * dg_dxi
                    - 4 * dg_dxi * f1**3 * g**(-1)
                    + 4 * f1**3 * g**(-2) * dg_dxi
                )
                hess_f2[i, 0] = hess_f2[0, i]  # Symmetry

            # d2f2/dx_i^2 for i = 2,...,n
            for i in range(1, n):
                hess_f2[i, i] = (
                    2 * dg_dxi * (0.25 * pow_025 * f1 / g**2 + 4 * pow_4 * f1 / g**2)
                    - g * (
                        0.25 * pow_025 * f1 * 2 * g**(-3) * dg_dxi**2
                        + 4 * pow_4 * f1 * 2 * g**(-3) * dg_dxi**2
                    )
                )

            # d2f2/dx_i dx_j for i, j = 2,...,n, i != j.  Since g is linear
            for i in range(1, n):
                for j in range(i + 1, n):
                    hess_f2[i, j] = (
                        -g * (
                            0.25 * pow_025 * f1 * 2 * g**(-3) * dg_dxi**2
                            + 4 * pow_4 * f1 * 2 * g**(-3) * dg_dxi**2
                        )
                    )
                    hess_f2[j, i] = hess_f2[i, j]
        return hess_f2

    def evaluate_f(self, x):
        """
        Evaluate the objective functions.

        Args:
            x (numpy.ndarray): Decision vector.

        Returns:
            numpy.ndarray: The objective vector [f1(x), f2(x)].
        """
        return np.array([self.f1(x), self.f2(x)])

    def evaluate_gradients_f(self, x):
        """
        Evaluate the gradients of the objective functions.

        Args:
            x (numpy.ndarray): Decision vector.

        Returns:
            numpy.ndarray: The gradients of the objective functions.
        """
        return np.array([self.grad_f1(x), self.grad_f2(x)])

    def evaluate_hessians_f(self, x):
        """
        Evaluate the Hessians of the objective functions.

        Args:
            x (numpy.ndarray): Decision vector.

        Returns:
            numpy.ndarray: The Hessians of the objective functions.
        """
        return np.array([self.hess_f1(x), self.hess_f2(x)])

    def evaluate_g(self, z):
        """
        Evaluate the constraint function g.

        Args:
            z (numpy.ndarray): Decision vector for constraints.

        Returns:
            numpy.ndarray: The constraint vector [g1(z), g2(z)].
        """
        if self.g_type[0] == 'zero':
            return np.zeros(self.m)
        elif self.g_type[0] == 'L1':
            res = np.zeros(self.m)
            for i in range(self.m):
                res[i] = np.linalg.norm((z-self.l1_shifts[i])*self.l1_ratios[i], ord=1)
            return res
        elif self.g_type[0] == 'indicator':
            # Example indicator constraint: check if z is within a certain region
            if 'lower_bound' not in self.g_type[1] or 'upper_bound' not in self.g_type[1]:
                raise ValueError("For indicator constraint, 'lower_bound' and 'upper_bound' must be specified in g_type[1].")
            lower_bound = self.g_type[1]['lower_bound']
            upper_bound = self.g_type[1]['upper_bound']
            violation = np.any(z < lower_bound) or np.any(z > upper_bound)
            return np.array([float(violation) * 1e10] * self.m) #Big violation if outside bounds
        elif self.g_type[0] == 'max':
             if 'value' not in self.g_type[1]:
                raise ValueError("For max constraint, 'value' must be specified in g_type[1].")
             value = self.g_type[1]['value']
             g_values = np.array([np.max(z) - value] * self.m)
             return g_values
        else:
            return np.zeros(self.m)  # Default to zero constraints

    def evaluate(self, x, z):
        """
        Evaluate the objective and constraint functions.

        Args:
            x (numpy.ndarray): Decision vector for objective functions.
            z (numpy.ndarray): Decision vector for constraint functions.

        Returns:
            numpy.ndarray: The combined objective and constraint vector.
        """
        f_values = self.evaluate_f(x)
        g_values = self.evaluate_g(z)
        return f_values + g_values

    def f_list(self):
        """
        Get the list of objective functions, gradients, and Hessians.

        Returns:
            list: A list of tuples, where each tuple contains the objective
                  function, its gradient, and its Hessian.
        """
        return [
            (self.f1, self.grad_f1, self.hess_f1),
            (self.f2, self.grad_f2, self.hess_f2)
        ]

