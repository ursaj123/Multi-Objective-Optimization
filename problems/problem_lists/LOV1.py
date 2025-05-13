import numpy as np
class LOV1:
    """
    LOV1 Problem

    This class defines the LOV1 multiobjective optimization problem.  The problem has
    two objective functions:

    f_1(x) = -1.05 * x_1^2 - 0.98 * x_2^2
    f_2(x) = -0.99 * (x_1 - 3)^2 - 1.03 * (x_2 - 2.5)^2

    The goal is to minimize both objective functions.

    Attributes:
        m (int): Number of objectives (fixed at 2).
        n (int): Number of variables (fixed at 2).
        lb (float): Lower bound for each variable.
        ub (float): Upper bound for each variable.
        bounds (tuple): A tuple of (lb, ub) pairs for each variable.
        constraints (list): A list of constraints (empty for LOV1).
        g_type (tuple):  A tuple specifying the type of function g(z) and its parameters.
        true_pareto_front (numpy.ndarray):  Approximation of the Pareto front.
        ref_point (numpy.ndarray):  Reference point for hypervolume calculation.
        num_samples (int): Number of samples used to approximate the Pareto front.
    """
    def __init__(self, n=2, lb=-5, ub=5, g_type=('zero', {}), num_samples=10000):
        """
        Initialize the LOV1 problem.

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
        self.constraints = []
        self.g_type = g_type
        self.num_samples = num_samples
        self.true_pareto_front = self.calculate_optimal_pareto_front()
        self.ref_point = np.max(self.true_pareto_front, axis=0) + 1e-4

    def calculate_optimal_pareto_front(self):
        """
        Approximate the Pareto front by sampling the decision space and
        finding the non-dominated objective vectors.

        Returns:
            numpy.ndarray: A 2D array representing the approximated Pareto front.
                           Each row corresponds to an objective vector.
        """
        samples = np.random.uniform(self.lb, self.ub, size=(self.num_samples, self.n))
        objective_values = np.array([self.evaluate_f(sample) for sample in samples])

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
        return -1.05 * x[0]**2 - 0.98 * x[1]**2

    def grad_f1(self, x):
        """
        Calculate the gradient of the first objective function.

        Args:
            x (numpy.ndarray): A 1D array representing the decision variables.

        Returns:
            numpy.ndarray: A 1D array representing the gradient.
        """
        return np.array([-2.1 * x[0], -1.96 * x[1]])

    def hess_f1(self, x):
        """
        Calculate the Hessian matrix of the first objective function.

        Args:
            x (numpy.ndarray): A 1D array representing the decision variables.

        Returns:
            numpy.ndarray: A 2D array representing the Hessian matrix.
        """
        return np.array([[-2.1, 0], [0, -1.96]])

    def f2(self, x):
        """
        Calculate the second objective function.

        Args:
            x (numpy.ndarray): A 1D array representing the decision variables.

        Returns:
            float: The value of the second objective function.
        """
        return -0.99 * (x[0] - 3)**2 - 1.03 * (x[1] - 2.5)**2

    def grad_f2(self, x):
        """
        Calculate the gradient of the second objective function.

        Args:
            x (numpy.ndarray): A 1D array representing the decision variables.

        Returns:
            numpy.ndarray: A 1D array representing the gradient.
        """
        return np.array([-1.98 * (x[0] - 3), -2.06 * (x[1] - 2.5)])

    def hess_f2(self, x):
        """
        Calculate the Hessian matrix of the second objective function.

        Args:
            x (numpy.ndarray): A 1D array representing the decision variables.

        Returns:
            numpy.ndarray: A 2D array representing the Hessian matrix.
        """
        return np.array([[-1.98, 0], [0, -2.06]])

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
            numpy.ndarray: A 2D array where each row is the gradient of an objective function.
        """
        return np.array([self.grad_f1(x), self.grad_f2(x)])

    def evaluate_hessians_f(self, x):
        """
        Evaluate the Hessian matrices of all objective functions.

        Args:
            x (numpy.ndarray): A 1D array representing the decision variables.

        Returns:
            numpy.ndarray: A 3D array where each 2D slice is the Hessian matrix of an objective function.
        """
        return np.array([self.hess_f1(x), self.hess_f2(x)])

    def evaluate_g(self, z):
        """
        Evaluate the constraint function g(z).  Currently only the 'zero' case is implemented.

        Args:
            z (numpy.ndarray): A 1D array representing the variables for the constraint function.

        Returns:
            numpy.ndarray: A 1D array containing the values of the constraint functions.

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
            x (numpy.ndarray): A 1D array representing the decision variables for the objective functions.
            z (numpy.ndarray): A 1D array representing the variables for the constraint functions.

        Returns:
            numpy.ndarray: A 1D array containing the combined values of the objective and constraint functions.
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
