import numpy as np

class LOV2:
    """
    LOV2 Problem

    This class defines the LOV2 multiobjective optimization problem. It has two
    objective functions:

    f_1(x) = -x_2
    f_2(x) = (x_2 - x_1^3) / (x_1 + 1)

    The goal is to minimize both objective functions.  Note that f_2(x) has a
    singularity at x_1 = -1.

    Attributes:
        m (int): Number of objectives (fixed at 2).
        n (int): Number of variables (fixed at 2).
        lb (float): Lower bound for each variable.
        ub (float): Upper bound for each variable.
        bounds (tuple): A tuple of (lb, ub) pairs for each variable.
        constraints (list): A list of constraints (x_1 != -1).
        g_type (tuple):  A tuple specifying the type of function g(z) and its parameters.
        true_pareto_front (numpy.ndarray): Approximation of the Pareto front.
        ref_point (numpy.ndarray): Reference point for hypervolume calculation.
        num_samples (int): Number of samples used to approximate the Pareto front.
    """
    def __init__(self, n=2, lb=-5, ub=5, g_type=('zero', {}), num_samples=10000):
        """
        Initialize the LOV2 problem.

        Args:
            n (int, optional): Number of variables. Defaults to 2.
            lb (float, optional): Lower bound for variables. Defaults to -5.
            ub (float, optional): Upper bound for variables. Defaults to 5.
            g_type (tuple, optional): Type of g function and its parameters.
                Defaults to ('zero', {}).
            num_samples (int, optional): Number of samples for Pareto front approximation.
                Defaults to 10000.
        """
        self.m = 2
        self.n = n
        self.lb = lb
        self.ub = ub
        self.bounds = tuple([(lb, ub) for _ in range(n)])
        self.constraints = [{'type': 'ineq', 'fun': lambda x: x[0] + 1 + 1e-8},  # x_1 > -1
                            {'type': 'ineq', 'fun': lambda x: -(x[0] + 1) + 1e-8}] # x_1 < -1
        self.g_type = g_type
        self.num_samples = num_samples
        self.true_pareto_front = self.calculate_optimal_pareto_front()
        self.ref_point = np.max(self.true_pareto_front, axis=0) + 1e-4

    def calculate_optimal_pareto_front(self):
        """
        Approximate the Pareto front by sampling the decision space and
        finding the non-dominated objective vectors.  This method handles the
        singularity at x_1 = -1.

        Returns:
            numpy.ndarray: A 2D array representing the approximated Pareto front.
                           Each row corresponds to an objective vector.
        """
        samples = np.random.uniform(self.lb, self.ub, size=(self.num_samples, self.n))
        # Filter out samples where x[0] is close to -1 to avoid singularity
        valid_samples = samples[np.abs(samples[:, 0] + 1) > 1e-2]
        objective_values = np.array([self.evaluate_f(sample) for sample in valid_samples])

        def is_dominated(point, front):
            """Check if a point is dominated by any point in the front."""
            for other in front:
                if np.all(other <= point) and np.any(other < point):
                    return True
            return False

        non_dominated_front = []
        for point in objective_values:
            if not is_dominated(point, non_dominated_front):
                non_dominated_front = [
                    other for other in non_dominated_front if not is_dominated(other, [point])
                ]
                non_dominated_front.append(point)
        return np.array(non_dominated_front)

    def f1(self, x):
        """
        Calculate the first objective function.

        Args:
            x (numpy.ndarray): A 1D array representing the decision variables.

        Returns:
            float: The value of the first objective function.
        """
        return -x[1]

    def grad_f1(self, x):
        """
        Calculate the gradient of the first objective function.

        Args:
            x (numpy.ndarray): A 1D array representing the decision variables.

        Returns:
            numpy.ndarray: A 1D array representing the gradient.
        """
        return np.array([0, -1])

    def hess_f1(self, x):
        """
        Calculate the Hessian matrix of the first objective function.

        Args:
            x (numpy.ndarray): A 1D array representing the decision variables.

        Returns:
            numpy.ndarray: A 2D array representing the Hessian matrix.
        """
        return np.array([[0, 0], [0, 0]])

    def f2(self, x):
        """
        Calculate the second objective function.  Handles the singularity at x_1 = -1
        by returning np.inf.

        Args:
            x (numpy.ndarray): A 1D array representing the decision variables.

        Returns:
            float: The value of the second objective function.
        """
        if abs(x[0] + 1) < 1e-9:
            return np.inf  # Handle the singularity
        return (x[1] - x[0]**3) / (x[0] + 1)

    def grad_f2(self, x):
        """
        Calculate the gradient of the second objective function.  Handles the
        singularity at x_1 = -1 by returning np.nan.

        Args:
            x (numpy.ndarray): A 1D array representing the decision variables.

        Returns:
            numpy.ndarray: A 1D array representing the gradient.
        """
        if abs(x[0] + 1) < 1e-9:
            return np.array([np.nan, np.nan])  # Handle the singularity
        num = x[1] - x[0]**3
        den = x[0] + 1
        d_num_dx = np.array([-3 * x[0]**2, 1])
        d_den_dx = np.array([1, 0])
        return (d_num_dx * den - num * d_den_dx) / (den**2)

    def hess_f2(self, x):
        """
        Calculate the Hessian matrix of the second objective function.
        Raises NotImplementedError as the calculation is complex.  Handles the
        singularity at x_1 = -1 by returning np.nan.

        Args:
            x (numpy.ndarray): A 1D array representing the decision variables.

        Returns:
            numpy.ndarray: A 2D array representing the Hessian matrix.

        Raises:
            NotImplementedError:  The hessian calculation is complex and not implemented.
        """
        if abs(x[0] + 1) < 1e-9:
            return np.array([[np.nan, np.nan], [np.nan, np.nan]])  # Handle singularity
        raise NotImplementedError("Hessian for f2 of LOV2 is not implemented yet.")

    def evaluate_f(self, x):
        """
        Evaluate all objective functions for a given decision vector.

        Args:
            x (numpy.ndarray): A 1D array representing the decision variables.

        Returns:
            numpy.ndarray: A 1D array containing the values of all objective functions.
        """
        return np.array([self.f1(x), self.f2(x)])

    def evaluate_gradients_f(self, x):
        """
        Evaluate the gradients of all objective functions.

        Args:
            x (numpy.ndarray): A 1D array representing the decision variables.

        Returns:
            numpy.ndarray: A 2D array where each row is the gradient of an objective
            function.
        """
        return np.array([self.grad_f1(x), self.grad_f2(x)])

    def evaluate_hessians_f(self, x):
        """
        Evaluate the Hessian matrices of all objective functions.

        Args:
            x (numpy.ndarray): A 1D array representing the decision variables.

        Returns:
            numpy.ndarray: A 3D array where each 2D slice is the Hessian matrix of an
            objective function.  Returns None if the Hessian for f2 is not implemented.
        """
        try:
            return np.array([self.hess_f1(x), self.hess_f2(x)])
        except NotImplementedError:
            print("Warning: Hessian for f2 of LOV2 is not implemented.")
            return None  # Or some other appropriate handling

    def evaluate_g(self, z):
        """
        Evaluate the constraint function g(z).  Currently only the 'zero' case is
        implemented.

        Args:
            z (numpy.ndarray): A 1D array representing the variables for the
            constraint function.

        Returns:
            numpy.ndarray: A 1D array containing the values of the constraint
            functions.

        Raises:
            ValueError: If an unknown g_type is provided.
            NotImplementedError: For 'L1', 'indicator', and 'max' g_types.
        """
        if self.g_type[0] == 'zero':
            return np.zeros(self.m)
        elif self.g_type[0] == 'L1':
            raise NotImplementedError("L1 norm for g is not implemented yet.")
        elif self.g_type[0] == 'indicator':
            raise NotImplementedError("Indicator function for g is not implemented yet.")
        elif self.g_type[0] == 'max':
            raise NotImplementedError("Max function for g is not implemented yet.")
        else:
            raise ValueError(f"Unknown g_type: {self.g_type[0]}")

    def evaluate(self, x, z):
        """
        Evaluate the objective functions and the constraint functions.

        Args:
            x (numpy.ndarray): A 1D array representing the decision variables for the
            objective functions.
            z (numpy.ndarray): A 1D array representing the variables for the
            constraint functions.

        Returns:
            numpy.ndarray: A 1D array containing the combined values of the
            objective and constraint functions.
        """
        f_values = self.evaluate_f(x)
        g_values = self.evaluate_g(z)
        return f_values + g_values

    def f_list(self):
        """
        Get a list of objective function tuples.  Each tuple contains the function,
        its gradient, and its Hessian.

        Returns:
            list: A list of tuples, where each tuple is (f_i, grad_f_i, hess_f_i).
        """
        return [
            (self.f1, self.grad_f1, self.hess_f1),
            (self.f2, self.grad_f2, self.hess_f2)
        ]
