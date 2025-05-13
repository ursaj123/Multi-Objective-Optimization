import numpy as np

class KW2:
    """
    KW2 Problem
    Formulation:
    J1 = 3(1 - x1)^2 * exp(-x1^2 - (x2 + 1)^2) - 10 * (x1/5 - x1^3 - x2^5) * exp(-x1^2 - x2^2) - 3 * exp(-(x1 + 2)^2 - x2^2) + 0.5 * (2x1 + x2)
    J2 = 3(1 + x2)^2 * exp(-x2^2 - (1 - x1)^2) - 10 * (-x2/5 + x2^3 + x1^5) * exp(-x1^2 - x2^2) - 3 * exp(-(2 - x2)^2 - x1^2)
    subject to -3 <= xi <= 3, i = 1, 2
    """
    def __init__(self, n=2, lb=-3, ub=3, g_type=('zero', {})):
        """
        Initialize the KW2 problem.

        Args:
            n (int): Number of variables.  (Although the problem is defined for n=2, I keep the argument for consistency)
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
        self.true_pareto_front = self.calculate_optimal_pareto_front() #Approximate
        self.ref_point = np.max(self.true_pareto_front, axis=0) + 1e-4

    def calculate_optimal_pareto_front(self, num_points=100):
        """
        Calculate an approximation of the Pareto front.  The Pareto front of this function is complex
        and cannot be easily expressed in a closed form.  This function provides a numerical approximation.

        Args:
            num_points (int): Number of points to generate for the approximation.

        Returns:
            numpy.ndarray: An approximation of the Pareto front as a 2D array.
        """
        # Since we don't have a closed-form solution, we'll generate points by varying x1 and x2.
        x1_values = np.linspace(self.lb, self.ub, num_points)
        x2_values = np.linspace(self.lb, self.ub, num_points)
        pareto_front_candidates = []

        # Evaluate J1 and J2 for all combinations of x1 and x2
        for x1 in x1_values:
            for x2 in x2_values:
                x = np.array([x1, x2])
                J1 = self.f1(x)
                J2 = self.f2(x)
                pareto_front_candidates.append([J1, J2])

        # Convert to numpy array
        candidates = np.array(pareto_front_candidates)

        # Apply non-dominated sorting to approximate the Pareto front.
        # This part assumes that lower values are better for both objectives, which is the
        # standard assumption in minimization.  The original problem is stated as maximization.
        # So, I'll need to negate the values.
        def is_dominated(candidate, others):
            for other in others:
                if (other[0] <= candidate[0] and other[1] <= candidate[1] and
                    (other[0] < candidate[0] or other[1] < candidate[1])):
                    return True
            return False

        pareto_front = []
        for candidate in candidates:
            if not is_dominated(candidate, candidates):
                pareto_front.append(candidate)
        return np.array(pareto_front)

    def f1(self, x):
        """
        Calculate the first objective function (J1).

        Args:
            x (numpy.ndarray): Decision vector (x1, x2).

        Returns:
            float: The value of the first objective function.
        """
        x1, x2 = x
        return (
            3 * (1 - x1)**2 * np.exp(-x1**2 - (x2 + 1)**2) -
            10 * (x1/5 - x1**3 - x2**5) * np.exp(-x1**2 - x2**2) -
            3 * np.exp(-(x1 + 2)**2 - x2**2) +
            0.5 * (2*x1 + x2)
        )

    def grad_f1(self, x):
        """
        Calculate the gradient of the first objective function (J1).

        Args:
            x (numpy.ndarray): Decision vector (x1, x2).

        Returns:
            numpy.ndarray: The gradient of the first objective function.
        """
        x1, x2 = x
        df1_dx1 = (
            -6 * (1 - x1) * np.exp(-x1**2 - (x2 + 1)**2) -
            2 * x1 * 3 * (1 - x1)**2 * np.exp(-x1**2 - (x2 + 1)**2) -
            10 * (1/5 - 3*x1**2) * np.exp(-x1**2 - x2**2) +
            10 * (x1/5 - x1**3 - x2**5) * 2*x1 * np.exp(-x1**2 - x2**2) +
            6 * (x1 + 2) * np.exp(-(x1 + 2)**2 - x2**2) +
            1
        )
        df1_dx2 = (
            -2 * (x2 + 1) * 3 * (1 - x1)**2 * np.exp(-x1**2 - (x2 + 1)**2) -
            10 * (-5*x2**4) * np.exp(-x1**2 - x2**2) +
            10 * (x1/5 - x1**3 - x2**5) * 2*x2 * np.exp(-x1**2 - x2**2) +
            6 * x2 * np.exp(-(x1 + 2)**2 - x2**2) +
            0.5
        )
        return np.array([df1_dx1, df1_dx2])

    def hess_f1(self, x):
        """
        Calculate the Hessian matrix of the first objective function (J1).

        Args:
            x (numpy.ndarray): Decision vector (x1, x2).

        Returns:
            numpy.ndarray: The Hessian matrix of the first objective function.
        """
        x1, x2 = x
        h11 = (
            6 * np.exp(-x1**2 - (x2 + 1)**2) +
            12 * x1 * (1 - x1) * np.exp(-x1**2 - (x2 + 1)**2) +
            4 * x1**2 * 3 * (1 - x1)**2 * np.exp(-x1**2 - (x2 + 1)**2) -
            60 * x1 * x1 * np.exp(-x1**2 - x2**2) -
            20 * (1/5 - 3 * x1**2) * x1 * np.exp(-x1**2 - x2**2) +
            20 * x1 * (x1/5 - x1**3 - x2**5) * np.exp(-x1**2 - x2**2) +
            4 * x1**2 * 10 * (x1/5 - x1**3 - x2**5) * np.exp(-x1**2 - x2**2) -
            12 * np.exp(-(x1 + 2)**2 - x2**2) +
            24 * (x1 + 2)**2 * np.exp(-(x1 + 2)**2 - x2**2)
        )
        h12 = (
            12 * (x2 + 1) * (1 - x1) * x1 * np.exp(-x1**2 - (x2 + 1)**2) +
            20 * x2 * x1 * np.exp(-x1**2 - x2**2) +
            20 * x2 * (x1/5 - x1**3 - x2**5) * np.exp(-x1**2 - x2**2) +
            12 * x2 * (x1 + 2) * np.exp(-(x1 + 2)**2 - x2**2)
        )
        h21 = h12
        h22 = (
            6 * (1 - x1)**2 * np.exp(-x1**2 - (x2 + 1)**2) +
            12 * (x2 + 1)**2 * 3 * (1 - x1)**2 * np.exp(-x1**2 - (x2 + 1)**2) -
            200 * x2**3 * np.exp(-x1**2 - x2**2) +
            20 * x2 * (x1/5 - x1**3 - x2**5) * np.exp(-x1**2 - x2**2) +
            4 * x2**2 * 10 * (x1/5 - x1**3 - x2**5) * np.exp(-x1**2 - x2**2) +
            6 * np.exp(-(x1 + 2)**2 - x2**2) +
            12 * x2**2 * np.exp(-(x1 + 2)**2 - x2**2)
        )

        return np.array([[h11, h12], [h21, h22]])


    def f2(self, x):
        """
        Calculate the second objective function (J2).

        Args:
            x (numpy.ndarray): Decision vector (x1, x2).

        Returns:
            float: The value of the second objective function.
        """
        x1, x2 = x
        return (
            3 * (1 + x2)**2 * np.exp(-x2**2 - (1 - x1)**2) -
            10 * (-x2/5 + x2**3 + x1**5) * np.exp(-x1**2 - x2**2) -
            3 * np.exp(-(2 - x2)**2 - x1**2)
        )

    def grad_f2(self, x):
        """
        Calculate the gradient of the second objective function (J2).

        Args:
            x (numpy.ndarray): Decision vector (x1, x2).

        Returns:
            numpy.ndarray: The gradient of the second objective function.
        """
        x1, x2 = x
        df2_dx1 = (
            6 * (1 + x2)**2 * (1 - x1) * np.exp(-x2**2 - (1 - x1)**2) -
            10 * (5*x1**4) * np.exp(-x1**2 - x2**2) +
            10 * (-x2/5 + x2**3 + x1**5) * 2*x1 * np.exp(-x1**2 - x2**2) +
            6 * x1 * np.exp(-(2 - x2)**2 - x1**2)
        )

        df2_dx2 = (
            6 * (1 + x2) * np.exp(-x2**2 - (1 - x1)**2) -
            2 * x2 * 3 * (1 + x2)**2 * np.exp(-x2**2 - (1 - x1)**2) -
            10 * (-1/5 + 3*x2**2) * np.exp(-x1**2 - x2**2) +
            10 * (-x2/5 + x2**3 + x1**5) * 2*x2 * np.exp(-x1**2 - x2**2) +
            6 * (2 - x2) * np.exp(-(2 - x2)**2 - x1**2)
        )
        return np.array([df2_dx1, df2_dx2])

    def hess_f2(self, x):
        """
        Calculate the Hessian matrix of the second objective function (J2).

        Args:
            x (numpy.ndarray): Decision vector (x1, x2).

        Returns:
            numpy.ndarray: The Hessian matrix of the second objective function.
        """
        x1, x2 = x
        h11 = (
            -6 * (1 + x2)**2 * np.exp(-x2**2 - (1 - x1)**2) +
            12 * (1 + x2)**2 * (1 - x1)**2 * np.exp(-x2**2 - (1 - x1)**2) -
            200 * x1**3 * np.exp(-x1**2 - x2**2) +
            20 * x1 * (-x2/5 + x2**3 + x1**5) * np.exp(-x1**2 - x2**2) +
            4 * x1**2 * 10 * (-x2/5 + x2**3 + x1**5) * np.exp(-x1**2 - x2**2) +
            6 * np.exp(-(2 - x2)**2 - x1**2) -
            12 * x1**2 * np.exp(-(2 - x2)**2 - x1**2)
        )

        h12 = (
            -12 * (1 + x2) * (1 - x1) * x1 * np.exp(-x2**2 - (1 - x1)**2) +
            20 * x1 * x2 * np.exp(-x1**2 - x2**2) +
            20 * x1 * (-x2/5 + x2**3 + x1**5) * np.exp(-x1**2 - x2**2) +
            12 * x1 * (2 - x2) * np.exp(-(2 - x2)**2 - x1**2)
        )
        h21 = h12

        h22 = (
            6 * np.exp(-x2**2 - (1 - x1)**2) -
            12 * x2 * (1 + x2) * np.exp(-x2**2 - (1 - x1)**2) +
            4 * x2**2 * 3 * (1 + x2)**2 * np.exp(-x2**2 - (1 - x1)**2) -
            60 * x2 * x2 * np.exp(-x1**2 - x2**2) -
            20 * (-1/5 + 3 * x2**2) * x2 * np.exp(-x1**2 - x2**2) +
            20 * x2 * (-x2/5 + x2**3 + x1**5) * np.exp(-x1**2 - x2**2) +
            4 * x2**2 * 10 * (-x2/5 + x2**3 + x1**5) * np.exp(-x1**2 - x2**2) -
            6 * np.exp(-(2 - x2)**2 - x1**2) +
            12 * (2 - x2)**2 * np.exp(-(2 - x2)**2 - x1**2)
        )
        return np.array([[h11, h12], [h21, h22]])

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
            # Example L1 norm constraint: sum of absolute values <= some_value
            if 'value' not in self.g_type[1]:
                raise ValueError("For L1 constraint, 'value' must be specified in g_type[1].")
            value = self.g_type[1]['value']
            g_values = np.array([np.sum(np.abs(z)) - value] * self.m)  # Repeat for each objective
            return g_values
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


