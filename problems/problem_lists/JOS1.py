import numpy as np

class JOS1:
    def __init__(self, n=5, lb=-3, ub = 3, g_type = ('zero', {}), fact=1):
        r"""
        JOS1 Problem
        f_1(x) = 0.5 * \|x\|^2
        f_2(x) = 0.5 * \|x - 2\|^2

        g_1(z) = zero/L1/indicator/max
        g_2(z) = zero/L1/indicator/max

        x and z are kept different for the sake of generality over algorithms (mainly for ADMM, but in 
        ADMM also, we are looking for the cases where x=z, so constraints are defined mainly for x)


        Arguments:
        m: Number of objectives
        n: Number of variables
        ub: Upper bound for variables
        g_type: Type of g function
        It can be one of ('zero', 'L1', 'indicator', 'max')
        Its parameters must be defined in the dictionary

        Defined Vars:
        bounds: Bounds for variables [(low, high), ...]

        constraints: List of constraints for the problem, these must be defined as per scipy optimize
        like {'type': 'ineq', 'fun': lambda x: x[0] - 1}.
        """

        self.m = 2  # Number of objectives
        self.n = n
        self.bounds = [(-5, 5) for _ in range(n)] # Assuming 2 variables for JOS1
        self.constraints = []
        self.g_type = g_type
        


        self.l1_ratios, self.l1_shifts = [], []
        if self.g_type[0]=='L1':
            for i in range(self.m):
                self.l1_ratios.append(1/((i+1)*self.n*fact))
                self.l1_shifts.append(i)
            self.l1_ratios = np.array(self.l1_ratios)
            self.l1_shifts = np.array(self.l1_shifts)
            

        # print(self.bounds)

    def feasible_space(self):
        test_x = np.random.uniform(self.bounds[0][0], self.bounds[0][1], size=(50000, self.n))
        f_values = np.array([self.evaluate(x, x) for x in test_x])
        return f_values
    

    def f1(self, x):
        return 0.5 * np.sum(x**2)

    def grad_f1(self, x):
        return x

    def f2(self, x):
        return 0.5 * np.sum((x - 2)**2)

    def grad_f2(self, x):
        return x - 2

    def hess_f1(self, x):
        return np.eye(self.n)

    def hess_f2(self, x):
        return np.eye(self.n)

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
            res = np.zeros(self.m)
            for i in range(self.m):
                res[i] = np.linalg.norm((z-self.l1_shifts[i])*self.l1_ratios[i], ord=1)
            return res
        elif self.g_type[0] == 'indicator':
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