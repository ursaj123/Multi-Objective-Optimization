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
    def __init__(self, n=2, lb=-4, ub=4, g_type=('zero', {}), fact=1):
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
        self.bounds = [(lb, ub) for _ in range(n)]
        self.constraints = []
        self.g_type = g_type
        

        self.l1_ratios, self.l1_shifts = [], []
        if self.g_type[0]=='L1':
            for i in range(self.m):
                self.l1_ratios.append(1/((i+1)*self.n*fact))
                self.l1_shifts.append(i)
            self.l1_ratios = np.array(self.l1_ratios)
            self.l1_shifts = np.array(self.l1_shifts)

    def feasible_space(self):
        test_x = np.random.uniform(self.bounds[0][0], self.bounds[0][1], size=(50000, self.n))
        f_values = np.array([self.evaluate(x, x) for x in test_x])
        return f_values

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
            res = np.zeros(self.m)
            for i in range(self.m):
                res[i] = np.linalg.norm((z-self.l1_shifts[i])*self.l1_ratios[i], ord=1)
            return res
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
