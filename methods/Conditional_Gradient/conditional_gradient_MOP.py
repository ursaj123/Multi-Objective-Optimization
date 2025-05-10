import numpy as np
from scipy.optimize import minimize
import sys
import os
sys.path.append('../problems')
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

class VectorOptimizationProblem:
    def __init__(self, m, n, objectives, gradients, bounds, constraints=None):
        self.m = m  # Number of objectives
        self.n = n  # Number of variables
        self.objectives = objectives  # List of objective functions
        self.gradients = gradients    # List of gradient functions
        self.bounds = bounds          # Bounds for variables [(low, high), ...]
        self.constraints = constraints if constraints is not None else []

    def evaluate(self, x):
        """Evaluate all objectives at x"""
        return np.array([f(x) for f in self.objectives])
    
    def evaluate_gradients(self, x):
        """Evaluate gradients of all objectives at x"""
        return [grad(x) for grad in self.gradients]

class ConditionalGradientOptimizer:
    def __init__(self, problem, step_size_strategy='armijo', beta=1e-4, delta=0.5, L=None, eta_max=0.9):
        self.problem = problem
        self.strategy = step_size_strategy
        self.beta = beta
        self.delta = delta
        self.L = L  # Required for adaptive strategy
        self.eta_max = eta_max
        self.D = None  # For nonmonotone strategy
        self.history = []

    def solve_subproblem(self, x):
        """Solve min max <∇f_i(x), s-x> subject to s ∈ Ω"""
        m, n = self.problem.m, self.problem.n
        grads = self.problem.evaluate_gradients(x)
        
        # Objective: min γ where γ ≥ <∇f_i, s-x> ∀i
        def obj(vars):
            return vars[-1]

        # Constraints: <grad_i, s-x> ≤ γ for each i
        constraints = []
        for i in range(m):
            grad_i = grads[i]
            def con_i(vars, grad_i=grad_i):
                s = vars[:-1]
                gamma = vars[-1]
                return grad_i @ (s - x) - gamma
            constraints.append({'type': 'ineq', 'fun': lambda v: -con_i(v)})

        # Add problem constraints (Ω) on s
        for con in self.problem.constraints:
            new_con = {'type': con['type']}
            if 'fun' in con:
                new_con['fun'] = lambda v, con=con: con['fun'](v[:-1])
            if 'jac' in con:
                new_con['jac'] = lambda v, con=con: np.append(con['jac'](v[:-1]), 0)
            constraints.append(new_con)

        # Bounds: s variables + gamma (unbounded)
        vars_bounds = self.problem.bounds + [(None, None)]

        # Initial point: s = x, γ = max(grad_i @ 0)
        vars0 = np.hstack([x, 0])

        res = minimize(obj, vars0, method='SLSQP', bounds=vars_bounds, constraints=constraints)
        if not res.success:
            raise RuntimeError(f"Subproblem failed: {res.message}")
        
        s = res.x[:-1]
        v = res.x[-1]
        return s, v

    def compute_step_size(self, x, d, v):
        if self.strategy == 'armijo':
            return self.armijo(x, d)
        elif self.strategy == 'adaptive':
            return self.adaptive(x, d, v)
        elif self.strategy == 'nonmonotone':
            return self.nonmonotone(x, d)
        else:
            raise ValueError("Invalid step size strategy")

    def armijo(self, x, d):
        """Armijo line search"""
        t = 1.0
        F_x = self.problem.evaluate(x)
        Jf_d = np.array([g @ d for g in self.problem.evaluate_gradients(x)])
        
        while True:
            x_new = x + t*d
            F_new = self.problem.evaluate(x_new)
            if np.all(F_new <= F_x + self.beta * t * Jf_d + 1e-12):
                return t
            t *= self.delta
            if t < 1e-12:
                return 0.0

    def adaptive(self, x, d, v):
        """Adaptive step size using L"""
        if self.L is None:
            raise ValueError("L must be provided for adaptive strategy")
        norm_d = np.linalg.norm(d)**2
        t = min(1.0, -v / (self.L * norm_d)) if norm_d > 1e-12 else 1.0
        return t

    def nonmonotone(self, x, d):
        """Nonmonotone line search"""
        if not self.history:
            self.D = self.problem.evaluate(x)
            self.history.append(self.D)
            eta = 0.0
        else:
            eta = min(self.eta_max, 0.5 / (len(self.history) + 1))  # Example eta update
            self.D = eta * self.D + (1 - eta) * self.history[-1]
        
        t = 1.0
        F_x = self.problem.evaluate(x)
        Jf_d = np.array([g @ d for g in self.problem.evaluate_gradients(x)])
        
        while True:
            x_new = x + t*d
            F_new = self.problem.evaluate(x_new)
            if np.all(F_new <= self.D + self.beta * t * Jf_d + 1e-12):
                self.history.append(F_new)
                return t
            t *= self.delta
            if t < 1e-12:
                return 0.0

    def optimize(self, x0, max_iter=100, tol=1e-6):
        x = np.array(x0, dtype=float)
        for _ in range(max_iter):
            s, v = self.solve_subproblem(x)
            if abs(v) < tol:
                break
            d = s - x
            t = self.compute_step_size(x, d, v)
            x += t*d
        return x

# Example usage with JOS1 problem
if __name__ == "__main__":
    # Define JOS1 problem
    m, n = 2, 4
    bounds = [(-5, 5)] * n
    
    def f1(x): return 0.5 * np.sum(x**2)
    def f2(x): return 0.5 * np.sum((x - 2)**2)
    def grad_f1(x): return x
    def grad_f2(x): return x - 2

    f1_eval, f2_eval = [], []
    for i in range(50):
        problem = VectorOptimizationProblem(
            m=2, n=n,
            objectives=[f1, f2],
            gradients=[grad_f1, grad_f2],
            bounds=bounds
        )
        
        # Initialize optimizer
        optimizer = ConditionalGradientOptimizer(problem, step_size_strategy='nonmonotone')
        
        # Run optimization from a random initial point
        x0 = np.random.uniform(-5, 5, n)
        x_opt = optimizer.optimize(x0)
        
        
        print("Optimized solution:", x_opt)
        print("Objective values:", problem.evaluate(x_opt))

        f1_eval.append(f1(x_opt))
        f2_eval.append(f2(x_opt))


    plt.scatter(f1_eval, f2_eval)
    plt.xlabel('f1')
    plt.ylabel('f2')
    
    plt.savefig(f'{os.getcwd()}/JOS1.png')
    