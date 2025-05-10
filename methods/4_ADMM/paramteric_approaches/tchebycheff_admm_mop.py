import numpy as np
from scipy.optimize import minimize

class TchebycheffADMM:
    def __init__(self, problem, n_weights=10, rho=1.0, max_iter=100, tol=1e-6):
        self.problem = problem  # MOProblem instance (e.g., SchafferProblem)
        self.n_weights = n_weights
        self.rho = rho
        self.max_iter = max_iter
        self.tol = tol
        
        # Generate weights and initialize variables
        self.weights = self._generate_weights()
        self.n = problem.n_vars
        self.m = problem.n_objs
        
        # Compute ideal point z_star = [min f1, min f2, ...]
        self.z_star = self._compute_ideal_point()
        
        # Initialize ADMM variables
        self.x = [np.random.uniform(*problem.bounds.T) for _ in range(n_weights)]
        self.z = np.mean(self.x, axis=0)
        self.y = [np.zeros(self.n) for _ in range(n_weights)]
        self.history = {'primal': [], 'dual': []}

    def _generate_weights(self):
        """Generate weights uniformly for two objectives."""
        return np.array([[i/(self.n_weights-1), 1 - i/(self.n_weights-1)] 
                        for i in range(self.n_weights)])
        # weights = np.zeros((self.n_weights, self.m))

    # def soft_threshold(x, threshold):
    #     r"""Soft-thresholding operator for L1 norm (element-wise)."""
    #     return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

    def _compute_ideal_point(self):
        """Compute z_i* = min f_i(x)."""
        z_star = np.zeros(self.m)
        for i in range(self.m):
            res = minimize(lambda x: self.problem.evaluate(x)[i], 
                           x0=np.mean(self.problem.bounds, axis=1),
                           bounds=self.problem.bounds)
            z_star[i] = res.fun
        return z_star

    def _solve_tchebycheff_subproblem(self, w_j, x_init, z, y_j):
        """Solve Tchebycheff + ADMM penalty subproblem."""
        def objective(x):
            f = self.problem.evaluate(x)
            t = np.max(w_j * (f - self.z_star))
            penalty = (self.rho / 2) * np.sum((x - z + y_j / self.rho)**2)
            return t + penalty
        res = minimize(objective, x_init, method='SLSQP', 
                       bounds=self.problem.bounds)
        return res.x

    def solve(self):
        """Run ADMM iterations with full Lagrangian terms."""
        for it in range(self.max_iter):
            # 1. Update all x_j in parallel
            new_x = []
            for j in range(self.n_weights):
                x_j = self._solve_tchebycheff_subproblem(
                    self.weights[j], self.x[j], self.z, self.y[j]
                )
                new_x.append(x_j)
            self.x = new_x
            
            # 2. Update consensus variable z
            z_old = self.z.copy()
            self.z = np.mean([x_j + y_j / self.rho 
                             for x_j, y_j in zip(self.x, self.y)], axis=0)
            
            # 3. Update dual variables y_j
            for j in range(self.n_weights):
                self.y[j] += self.rho * (self.x[j] - self.z)
            
            # Check convergence
            primal_res = np.sqrt(sum(np.linalg.norm(x_j - self.z)**2 
                                  for x_j in self.x))
            dual_res = self.rho * np.sqrt(self.n_weights) * np.linalg.norm(self.z - z_old)
            
            self.history['primal'].append(primal_res)
            self.history['dual'].append(dual_res)
            
            if primal_res < self.tol and dual_res < self.tol:
                print(f"Converged at iteration {it}")
                break
        return self.x