import numpy as np

class DUMMY1:
    def __init__(self, n=1, lb=-5, ub = 5, g_type = ('zero', {})):
        r"""
        JOS1 Problem
        f_1(x) = x**2
        f_2(x) = (x-2)**2

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
        self.lb = lb
        self.ub = ub
        self.bounds = [(lb, ub) for _ in range(n)] # Assuming 2 variables for JOS1
        self.constraints = []
        self.g_type = g_type
        self.true_pareto_front = self.calculate_optimal_pareto_front()
        # print(f"true_pareto_front: {self.true_pareto_front}")
        self.ref_point = np.max(self.true_pareto_front, axis=0)
        # print(f"ref_point: {self.ref_point}")

        # print(self.bounds)

    def calculate_optimal_pareto_front(self):
        """
        These are the values of the objectives at the true pareto front
        f_1(x) = 0.5 * \|x\|^2
        f_2(x) = 0.5 * \|x - 2\|^2
        """
        x = np.linspace(0, 2, 100)
        list_ = []
        for i in x:
            list_.append(self.evaluate(i, i))
        return np.array(list_)
    

    def f1(self, x):
        return x**2

    def grad_f1(self, x):
        return 2*x

    def f2(self, x):
        return (x - 2)**2

    def grad_f2(self, x):
        return 2*(x - 2)

    def hess_f1(self, x):
        return 2*np.eye(self.n)

    def hess_f2(self, x):
        return 2*np.eye(self.n)

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