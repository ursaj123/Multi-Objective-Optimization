import numpy as np

class AP4:
    r"""
    AP4 Problem (3D Multiobjective Optimization)
    
    A vector optimization problem: minₓ (F₁(x), F₂(x), F₃(x)) where:
    - F₁(x) = 1/9[(x₁-1)⁴ + 2(x₂-2)⁴ + 3(x₃-3)⁴]
    - F₂(x) = exp((x₁+x₂+x₃)/3) + x₁² + x₂² + x₃²
    - F₃(x) = 1/12[3e^{-x₁} + 4e^{-x₂} + 3e^{-x₃}]
    
    Parameters:
    n : int
        Number of variables (fixed at 3)
    lb : float
        Lower bound for all variables
    ub : float
        Upper bound for all variables
    g_type : tuple
        Type of non-smooth term G(z), format: (type_str, params_dict)
    
    Attributes:
    bounds : list
        Variable bounds [(lb, ub), ...]
    constraints : list
        Constraints list (empty for this problem)
    true_pareto_front : ndarray
        Approximate Pareto front points
    ref_point : ndarray
        Reference point for hypervolume calculation
    """
    
    def __init__(self, n=3, lb=-2, ub=2, g_type=('zero', {}), fact=1):
        self.m = 3  # Number of objectives
        self.n = n  # Number of variables (fixed at 3)
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
        r"""F₁(x) = 1/9[(x₁-1)⁴ + 2(x₂-2)⁴ + 3(x₃-3)⁴]"""
        return (1/9) * ((x[0] - 1)**4 + 2*(x[1] - 2)**4 + 3*(x[2] - 3)**4)

    def grad_f1(self, x):
        r"""∇F₁(x) = [4/9(x₁-1)³, 8/9(x₂-2)³, 12/9(x₃-3)³]"""
        return np.array([
            (4/9) * (x[0] - 1)**3,
            (8/9) * (x[1] - 2)**3,
            (12/9) * (x[2] - 3)**3
        ])

    def hess_f1(self, x):
        r"""Hess(F₁) = diag(12/9(x₁-1)², 24/9(x₂-2)², 36/9(x₃-3)²)"""
        return np.diag([
            (12/9) * (x[0] - 1)**2,
            (24/9) * (x[1] - 2)**2,
            (36/9) * (x[2] - 3)**2
        ])

    def f2(self, x):
        r"""F₂(x) = exp((x₁+x₂+x₃)/3) + x₁² + x₂² + x₃²"""
        return np.exp(sum(x)/3) + np.sum(x**2)

    def grad_f2(self, x):
        r"""∇F₂(x) = [e^{S}/3 + 2x₁, e^{S}/3 + 2x₂, e^{S}/3 + 2x₃] where S = (Σx_i)/3"""
        exp_term = np.exp(sum(x)/3)/3
        return np.array([exp_term + 2*x[i] for i in range(3)])

    def hess_f2(self, x):
        r"""
        Hess(F₂) = [[e^S/9 + 2, e^S/9,    e^S/9   ],
                   [e^S/9,    e^S/9 + 2, e^S/9   ],
                   [e^S/9,    e^S/9,    e^S/9 + 2]]
        where S = (Σx_i)/3
        """
        exp_term = np.exp(sum(x)/3)/9
        hess = np.full((3, 3), exp_term)
        np.fill_diagonal(hess, hess.diagonal() + 2)
        return hess

    def f3(self, x):
        r"""F₃(x) = 1/12[3e^{-x₁} + 4e^{-x₂} + 3e^{-x₃}]"""
        return (3*np.exp(-x[0]) + 4*np.exp(-x[1]) + 3*np.exp(-x[2]))/12

    def grad_f3(self, x):
        r"""∇F₃(x) = [-3e^{-x₁}/12, -4e^{-x₂}/12, -3e^{-x₃}/12]"""
        return np.array([-3*np.exp(-x[0])/12, 
                        -4*np.exp(-x[1])/12,
                        -3*np.exp(-x[2])/12])

    def hess_f3(self, x):
        r"""Hess(F₃) = diag(3e^{-x₁}/12, 4e^{-x₂}/12, 3e^{-x₃}/12)"""
        return np.diag([3*np.exp(-x[0])/12,
                       4*np.exp(-x[1])/12,
                       3*np.exp(-x[2])/12])

    def evaluate_f(self, x):
        """Evaluate all objectives at x"""
        return np.array([self.f1(x), self.f2(x), self.f3(x)])

    def evaluate_gradients_f(self, x):
        """Return gradients for all objectives as (3, 3) array"""
        return np.array([self.grad_f1(x), self.grad_f2(x), self.grad_f3(x)])

    def evaluate_hessians_f(self, x):
        """Return Hessians for all objectives as (3, 3, 3) array"""
        return np.array([self.hess_f1(x), self.hess_f2(x), self.hess_f3(x)])

    def evaluate_g(self, z):
        """Evaluate non-smooth term G(z) (default: zero vector)"""
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
        """Evaluate combined objective F(x) + G(z)"""
        return self.evaluate_f(x) + self.evaluate_g(z)

    def f_list(self):
        """Return list of (objective, gradient, hessian) tuples"""
        return [
            (self.f1, self.grad_f1, self.hess_f1),
            (self.f2, self.grad_f2, self.hess_f2),
            (self.f3, self.grad_f3, self.hess_f3)
        ]