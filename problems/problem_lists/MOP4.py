import numpy as np

class MOP4:
    def __init__(self, g_type=('zero', {})):
        """
        MOP4 Problem

        f_1(x1, x2, x3) = sum_{i=1}^{2} -10 exp(-0.2 sqrt(xi^2 + x_{i+1}^2))
        f_2(x1, x2, x3) = sum_{i=1}^{3} |xi|^0.8 + 5 sin(xi^3)

        Arguments:
        g_type: Type of g function
        """
        self.n = 3
        self.m = 2
        self.bounds = tuple([(-5, 5)] * 3)
        self.lb = -5
        self.ub = 5
        self.g_type = g_type
        self.constraints = []
        # Approximate reference point.  The minimum of f1 is -20, and the minimum of f2 is around -4
        self.true_pareto_front = self.calculate_optimal_pareto_front()
        self.ref_point = np.max(self.true_pareto_front, axis=0) + 1e-4

    def f1(self, x):
        return np.sum(-10 * np.exp(-0.2 * np.sqrt(x[0]**2 + x[1]**2)), axis=-1) + (-10 * np.exp(-0.2 * np.sqrt(x[1]**2 + x[2]**2)))

    def grad_f1(self, x):
        term1_exp = np.exp(-0.2 * np.sqrt(x[0]**2 + x[1]**2))
        term2_exp = np.exp(-0.2 * np.sqrt(x[1]**2 + x[2]**2))
        sqrt_term1 = np.sqrt(x[0]**2 + x[1]**2)
        sqrt_term2 = np.sqrt(x[1]**2 + x[2]**2)

        grad_f1_x1 = -10 * term1_exp * (-0.2) * (0.5 * (2*x[0]) / sqrt_term1)
        grad_f1_x2 = -10 * term1_exp * (-0.2) * (0.5 * (2*x[1]) / sqrt_term1) +  -10 * term2_exp * (-0.2) * (0.5 * (2*x[1]) / sqrt_term2)
        grad_f1_x3 = -10 * term2_exp * (-0.2) * (0.5 * (2*x[2]) / sqrt_term2)

        return np.array([grad_f1_x1, grad_f1_x2, grad_f1_x3])

    def hess_f1(self, x):
        term1_exp = np.exp(-0.2 * np.sqrt(x[0]**2 + x[1]**2))
        term2_exp = np.exp(-0.2 * np.sqrt(x[1]**2 + x[2]**2))
        sqrt_term1 = np.sqrt(x[0]**2 + x[1]**2)
        sqrt_term2 = np.sqrt(x[1]**2 + x[2]**2)

        d2f1_dx12 = -10 * term1_exp * (
            (-0.2) * (1/sqrt_term1 - (x[0]**2) / (sqrt_term1**3)) + (-0.2 * (x[0] / sqrt_term1))**2
        )
        d2f1_dx22_1 = -10 * term1_exp * (
            (-0.2) * (1/sqrt_term1 - (x[1]**2) / (sqrt_term1**3)) + (-0.2 * (x[1] / sqrt_term1))**2
        )
        d2f1_dx22_2 = -10 * term2_exp * (
            (-0.2) * (1/sqrt_term2 - (x[1]**2) / (sqrt_term2**3)) + (-0.2 * (x[1] / sqrt_term2))**2
        )
        d2f1_dx32 = -10 * term2_exp * (
            (-0.2) * (1/sqrt_term2 - (x[2]**2) / (sqrt_term2**3)) + (-0.2 * (x[2] / sqrt_term2))**2
        )

        d2f1_dx1dx2 = -10 * term1_exp * (
            (-0.2) * (-(x[0] * x[1]) / (sqrt_term1**3)) + (-0.2 * (x[0] / sqrt_term1)) * (-0.2 * (x[1] / sqrt_term1))
        )
        d2f1_dx2dx3 = -10 * term2_exp * (
            (-0.2) * (-(x[1] * x[2]) / (sqrt_term2**3)) + (-0.2 * (x[1] / sqrt_term2)) * (-0.2 * (x[2] / sqrt_term2))
        )
        d2f1_dx3dx1 = 0


        H = np.array([
            [d2f1_dx12, d2f1_dx1dx2, d2f1_dx3dx1],
            [d2f1_dx1dx2, d2f1_dx22_1 + d2f1_dx22_2, d2f1_dx2dx3],
            [d2f1_dx3dx1, d2f1_dx2dx3, d2f1_dx32]
        ])
        return H


    def f2(self, x):
        return np.sum(np.abs(x)**0.8 + 5 * np.sin(x**3), axis=-1)

    def grad_f2(self, x):
        grad_f2_x = 0.8 * np.sign(x) * np.abs(x)**(-0.2) + 15 * (x**2) * np.cos(x**3)
        return grad_f2_x

    def hess_f2(self, x):
        hess_f2_x = (
            -0.16 * np.abs(x)**(-1.2) +
            90 * x * np.cos(x**3) -
            45 * (x**4) * np.sin(x**3)
        )
        return np.diag(hess_f2_x)


    def evaluate_f(self, x):
        return np.array([self.f1(x), self.f2(x)])

    def evaluate_gradients_f(self, x):
        return np.array([self.grad_f1(x), self.grad_f2(x)])

    def evaluate_hessians_f(self, x):
        return np.array([self.hess_f1(x), self.hess_f2(x)])

    def calculate_optimal_pareto_front(self, n_samples=100):
        # A rough approximation of the Pareto front.
        x1_values = np.linspace(-5, 5, n_samples)
        x2_values = np.linspace(-5, 5, n_samples)
        x3_values = np.linspace(-5, 5, n_samples)
        pareto_front = np.zeros((n_samples, 2))
        for i in range(n_samples):
            x = np.array([x1_values[i], x2_values[i], x3_values[i]])
            pareto_front[i, :] = self.evaluate_f(x)
        return pareto_front

    def evaluate_g(self, z):
        g_values = np.zeros(self.m)
        g_type, params = self.g_type
        if g_type == 'zero':
            return g_values
        elif g_type == 'L1':
            pass
        elif g_type == 'indicator':
            pass
        elif g_type == 'max':
            pass
        else:
            pass

    
    def evaluate(self, x, z):
        f_values = self.evaluate_f(x)
        g_values = self.evaluate_g(z)
        return f_values + g_values

    def f_list(self):
        return [
            (self.f1, self.grad_f1, self.hess_f1),
            (self.f2, self.grad_f2, self.hess_f2)
        ]
