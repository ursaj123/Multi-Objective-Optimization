import numpy as np
class MOP3:
    def __init__(self, g_type=('zero', {})):
        """
        MOP3 Problem

        Max. f_1(x1, x2) = -1 - (A1 - B1)^2 - (A2 - B2)^2
        Max. f_2(x1, x2) = -(x1 + 3)^2 - (x2 + 1)^2

        Where:
        A1 = 0.5 sin 1 - 2 cos 1 + sin 2 - 1.5 cos 2
        A2 = 1.5 sin 1 - cos 1 + 2 sin 2 - 0.5 cos 2
        B1 = 0.5 sin x1 - 2 cos x1 + sin x2 - 1.5 cos x2
        B2 = 1.5 sin x1 - cos x1 + 2 sin x2 - 0.5 cos x2

        Arguments:
        g_type: Type of g function
        """
        self.n = 2
        self.m = 2
        self.lb = -np.pi
        self.ub = np.pi
        self.bounds = [(self.lb, self.ub)] * 2
        self.g_type = g_type
        self.constraints = []
        self.A1 = 0.5 * np.sin(1) - 2 * np.cos(1) + np.sin(2) - 1.5 * np.cos(2)
        self.A2 = 1.5 * np.sin(1) - np.cos(1) + 2 * np.sin(2) - 0.5 * np.cos(2)
        # For maximization, we'll negate the objective functions.

        self.l1_ratios, self.l1_shifts = [], []
        if self.g_type[0]=='L1':
            for i in range(self.m):
                self.l1_ratios.append(1/((i+1)*self.n))
                self.l1_shifts.append(i)
            self.l1_ratios = np.array(self.l1_ratios)
            self.l1_shifts = np.array(self.l1_shifts)

    
    def feasible_space(self):
        test_x = np.random.uniform(self.bounds[0][0], self.bounds[0][1], size=(1000, self.n))
        f_values = np.array([self.evaluate(x, x) for x in test_x])
        return f_values

    def f1(self, x):
        B1 = 0.5 * np.sin(x[0]) - 2 * np.cos(x[0]) + np.sin(x[1]) - 1.5 * np.cos(x[1])
        B2 = 1.5 * np.sin(x[0]) - np.cos(x[0]) + 2 * np.sin(x[1]) - 0.5 * np.cos(x[1])
        return 1 + (self.A1 - B1)**2 + (self.A2 - B2)**2  # Negated for minimization

    def grad_f1(self, x):
        B1 = 0.5 * np.sin(x[0]) - 2 * np.cos(x[0]) + np.sin(x[1]) - 1.5 * np.cos(x[1])
        B2 = 1.5 * np.sin(x[0]) - np.cos(x[0]) + 2 * np.sin(x[1]) - 0.5 * np.cos(x[1])

        dB1_dx1 = 0.5 * np.cos(x[0]) + 2 * np.sin(x[0])
        dB1_dx2 = np.cos(x[1]) + 1.5 * np.sin(x[1])
        dB2_dx1 = 1.5 * np.cos(x[0]) + np.sin(x[0])
        dB2_dx2 = 2 * np.cos(x[1]) + 0.5 * np.sin(x[1])

        grad_f1_x1 = 2 * (self.A1 - B1) * (-dB1_dx1) + 2 * (self.A2 - B2) * (-dB2_dx1)
        grad_f1_x2 = 2 * (self.A1 - B1) * (-dB1_dx2) + 2 * (self.A2 - B2) * (-dB2_dx2)
        return np.array([grad_f1_x1, grad_f1_x2])

    def hess_f1(self, x):
        B1 = 0.5 * np.sin(x[0]) - 2 * np.cos(x[0]) + np.sin(x[1]) - 1.5 * np.cos(x[1])
        B2 = 1.5 * np.sin(x[0]) - np.cos(x[0]) + 2 * np.sin(x[1]) - 0.5 * np.cos(x[1])

        dB1_dx1 = 0.5 * np.cos(x[0]) + 2 * np.sin(x[0])
        dB1_dx2 = np.cos(x[1]) + 1.5 * np.sin(x[1])
        dB2_dx1 = 1.5 * np.cos(x[0]) + np.sin(x[0])
        dB2_dx2 = 2 * np.cos(x[1]) + 0.5 * np.sin(x[1])

        d2B1_dx12 = -0.5 * np.sin(x[0]) + 2 * np.cos(x[0])
        d2B1_dx22 = -np.sin(x[1]) + 1.5 * np.cos(x[1])
        d2B1_dx1dx2 = 0
        d2B2_dx12 = -1.5 * np.sin(x[0]) + np.cos(x[0])
        d2B2_dx22 = -2 * np.sin(x[1]) + 0.5 * np.cos(x[1])
        d2B2_dx1dx2 = 0

        H11 = 2 * ((-dB1_dx1)**2 - (self.A1 - B1) * d2B1_dx12 + (-dB2_dx1)**2 - (self.A2 - B2) * d2B2_dx12)
        H12 = 2 * ((-dB1_dx1) * (-dB1_dx2) - (self.A1 - B1) * d2B1_dx1dx2 + (-dB2_dx1) * (-dB2_dx2) - (self.A2 - B2) * d2B2_dx1dx2)
        H21 = H12
        H22 = 2 * ((-dB1_dx2)**2 - (self.A1 - B1) * d2B1_dx22 + (-dB2_dx2)**2 - (self.A2 - B2) * d2B2_dx22)

        return np.array([[H11, H12], [H21, H22]])


    def f2(self, x):
        return (x[0] + 3)**2 + (x[1] + 1)**2 # Negated for minimization

    def grad_f2(self, x):
        return np.array([2 * (x[0] + 3), 2 * (x[1] + 1)])

    def hess_f2(self, x):
        return np.array([[2, 0], [0, 2]])

    def evaluate_f(self, x):
        return np.array([self.f1(x), self.f2(x)])

    def evaluate_gradients_f(self, x):
        return np.array([self.grad_f1(x), self.grad_f2(x)])

    def evaluate_hessians_f(self, x):
        return np.array([self.hess_f1(x), self.hess_f2(x)])

    def calculate_optimal_pareto_front(self, n_samples=100):
        #  The Pareto front for MOP3 is obtained when x1 = -3 and x2 = -1
        return np.array([[-3.0, -1.0], [1.0, 2.0]])

    def evaluate_g(self, z):
        g_values = np.zeros(self.m)
        g_type, params = self.g_type
        if g_type == 'zero':
            return g_values
        elif g_type == 'L1':
            res = np.zeros(self.m)
            for i in range(self.m):
                res[i] = np.linalg.norm((z-self.l1_shifts[i])*self.l1_ratios[i], ord=1)
            return res
        elif g_type == 'indicator':
            pass
        elif g_type == 'max':
            pass

    def evaluate(self, x, z):
        f_values = self.evaluate_f(x)
        g_values = self.evaluate_g(z)
        # For MOP3, we need to negate the objective values because it's a maximization problem.
        return -f_values + g_values

    def f_list(self):
        return [
            (self.f1, self.grad_f1, self.hess_f1),
            (self.f2, self.grad_f2, self.hess_f2)
        ]
