import numpy as np

# class DGO1:
#     def __init__(self, lb=-10, ub=13, g_type=('zero', {})):
#         """
#         DGO1 Problem
#         f_1(x) = sin(x)
#         f_2(x) = sin(x + 0.7)

#         g_1(z) = zero/L1/indicator/max
#         g_2(z) = zero/L1/indicator/max

#         Arguments:
#         ub: Upper bound for the variable
#         lb: Lower bound for the variable
#         g_type: Type of g function
#                 It can be one of ('zero', 'L1', 'indicator', 'max')
#                 Its parameters must be defined in the dictionary

#         Defined Vars:
#         bounds: Bounds for the variable [(low, high)]
#         constraints: List of constraints (empty for this problem)
#         """
#         self.m = 2  # Number of objectives
#         self.n = 1  # Number of variables
#         self.lb = lb
#         self.ub = ub
#         self.bounds = tuple([(lb, ub)])
#         self.constraints = []
#         self.g_type = g_type
#         self.true_pareto_front = self.calculate_optimal_pareto_front()
#         self.ref_point = np.max(self.true_pareto_front, axis=0) + 1e-4

#     def calculate_optimal_pareto_front(self, num_points=100):
#         """
#         Approximates the Pareto front by sampling x values.
#         """
#         x_values = np.linspace(self.lb, self.ub, num_points)
#         f1_values = np.sin(x_values)
#         f2_values = np.sin(x_values + 0.7)
#         pareto_front = np.column_stack([f1_values, f2_values])
#         return pareto_front

#     def f1(self, x):
#         return np.sin(x)

#     def grad_f1(self, x):
#         return np.cos(x)

#     def f2(self, x):
#         return np.sin(x + 0.7)

#     def grad_f2(self, x):
#         return np.cos(x + 0.7)

#     def hess_f1(self, x):
#         return -np.sin(x)

#     def hess_f2(self, x):
#         return -np.sin(x + 0.7)

#     def evaluate_f(self, x):
#         return np.array([self.f1(x), self.f2(x)])

#     def evaluate_gradients_f(self, x):
#         return np.array([self.grad_f1(x), self.grad_f2(x)])

#     def evaluate_hessians_f(self, x):
#         return np.array([self.hess_f1(x), self.hess_f2(x)])

#     def evaluate_g(self, z):
#         if self.g_type[0] == 'zero':
#             return np.zeros(self.m)
#         elif self.g_type[0] == 'L1':
#             pass
#         elif self.g_type[0] == 'indicator':
#             pass
#         elif self.g_type[0] == 'max':
#             pass
#         return np.zeros(self.m)

#     def evaluate(self, x, z):
#         f_values = self.evaluate_f(x)
#         g_values = self.evaluate_g(z)
#         return f_values + g_values

#     def f_list(self):
#         return [
#             (self.f1, self.grad_f1, self.hess_f1),
#             (self.f2, self.grad_f2, self.hess_f2)
#         ]



class DGO1:
    def __init__(self, lb=-10, ub=13, g_type=('zero', {})):
        """
        DGO1 Problem
        f_1(x) = sin(x)
        f_2(x) = sin(x + 0.7)

        g_1(z) = zero/L1/indicator/max
        g_2(z) = zero/L1/indicator/max

        Arguments:
        ub: Upper bound for the variable
        lb: Lower bound for the variable
        g_type: Type of g function
                It can be one of ('zero', 'L1', 'indicator', 'max')
                Its parameters must be defined in the dictionary

        Defined Vars:
        bounds: Bounds for the variable [(low, high)]
        constraints: List of constraints (empty for this problem)
        """
        self.m = 2  # Number of objectives
        self.n = 1  # Number of variables
        self.lb = lb
        self.ub = ub
        self.bounds = tuple([(lb, ub)])
        self.constraints = []
        self.g_type = g_type
        self.true_pareto_front = self.calculate_optimal_pareto_front()
        self.ref_point = np.max(self.true_pareto_front, axis=0) + 1e-4

    def calculate_optimal_pareto_front(self, num_points=100):
        """
        Generates points on the Pareto front for DGO1 using the relationship
        between f1 and f2.
        """
        f1_values = np.linspace(-1, 1, num_points)
        cos_07 = np.cos(0.7)
        sin_07 = np.sin(0.7)
        f2_plus = f1_values * cos_07 + np.sqrt(1 - f1_values**2) * sin_07
        f2_minus = f1_values * cos_07 - np.sqrt(1 - f1_values**2) * sin_07
        pareto_front = np.column_stack([np.concatenate([f1_values, f1_values]),
                                         np.concatenate([f2_plus, f2_minus])])
        return pareto_front

    def f1(self, x):
        return np.sin(x)

    def grad_f1(self, x):
        return np.cos(x)

    def f2(self, x):
        return np.sin(x + 0.7)

    def grad_f2(self, x):
        return np.cos(x + 0.7)

    def hess_f1(self, x):
        return -np.sin(x)

    def hess_f2(self, x):
        return -np.sin(x + 0.7)

    def evaluate_f(self, x):
        return np.array([self.f1(x), self.f2(x)])

    def evaluate_gradients_f(self, x):
        return np.array([self.grad_f1(x), self.grad_f2(x)])

    def evaluate_hessians_f(self, x):
        return np.array([self.hess_f1(x), self.hess_f2(x)])

    def evaluate_g(self, z):
        if self.g_type[0] == 'zero':
            return np.zeros(self.m)
        elif self.g_type[0] == 'L1':
            pass
        elif self.g_type[0] == 'indicator':
            pass
        elif self.g_type[0] == 'max':
            pass
        return np.zeros(self.m)

    def evaluate(self, x, z):
        f_values = self.evaluate_f(x)
        g_values = self.evaluate_g(z)
        return f_values + g_values

    def f_list(self):
        return [
            (self.f1, self.grad_f1, self.hess_f1),
            (self.f2, self.grad_f2, self.hess_f2)
        ]



