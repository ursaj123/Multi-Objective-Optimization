# File: multi_objective_gradient.py
import numpy as np
from scipy.optimize import minimize
import time

class MultiObjectiveSteepestDescent:
    r"""
    Implements the steepest descent method for multi-objective optimization from:
    Fliege, J., & Svaiter, B. F. (2000). Steepest descent methods for multicriteria optimization.
    
    Parameters:
    problem : object
        Problem instance with required methods
    beta : float (default: 0.1)
        Armijo rule parameter
    sigma : float (default: 0.1)
        Direction solution tolerance
    tol : float (default: 1e-6)
        Pareto criticality tolerance
    max_iter : int (default: 1000)
        Maximum iterations
    
    Attributes:
    history : list
        Optimization path (decision space)
    f_history : list
        Optimization path (objective space)
    result : dict
        Final optimization result
    """
    
    def __init__(self, problem, beta=0.1, sigma=0.1, tol=1e-6, max_iter=1000):
        self.problem = problem
        self.beta = beta
        self.sigma = sigma
        self.tol = tol
        self.max_iter = max_iter
        self.history = []
        self.f_history = []
        self.result = {}

    def solve(self, x0):
        """
        Run optimization from initial point x0
        
        Returns:
        result : dict
            Contains:
            - 'x': final decision variables
            - 'f': final objective values
            - 'runtime': computation time
            - 'converged': bool indicating convergence
            - 'iterations': number of iterations
        """
        start_time = time.time()
        x = np.array(x0).astype(float)
        self.history = [x.copy()]
        self.f_history = [self.problem.evaluate_f(x)]
        
        converged = False
        
        for k in range(self.max_iter):
            # Compute Jacobian (stacked gradients)
            J = self.problem.evaluate_gradients_f(x)
            
            # Solve direction subproblem
            v, alpha = self._compute_direction(J)
            
            # Check Pareto criticality
            if alpha > -self.tol:
                converged = True
                break
                
            # Armijo line search
            t = self._line_search(x, v, J)
            
            # Update position with bounds
            x_new = self._project(x + t*v)
            
            # Store iteration
            x = x_new
            self.history.append(x.copy())
            self.f_history.append(self.problem.evaluate_f(x))
            
        self.result = {
            'x': x,
            'f': self.problem.evaluate_f(x),
            'runtime': time.time() - start_time,
            'converged': converged,
            'iterations': k+1
        }
        return self.result

    def _compute_direction(self, J):
        """Solve QP: min α + ½||v||² s.t. Jv ≤ α"""
        m, n = J.shape
        
        # Solve using scipy's QP solver
        def obj(z):
            return z[0] + 0.5 * z[1:].dot(z[1:])
            
        constraints = [{
            'type': 'ineq',
            'fun': lambda z, i=i: -J[i].dot(z[1:]) + z[0]
        } for i in range(m)]
        
        res = minimize(obj, x0=np.zeros(n+1), 
                      constraints=constraints, 
                      method='SLSQP',
                      options={'ftol': 1e-6})
        
        if not res.success:
            raise RuntimeError(f"Direction QP failed: {res.message}")
            
        return res.x[1:], res.x[0]

    def _line_search(self, x, v, J):
        """Armijo rule for vector optimization"""
        t = 1.0
        f0 = self.problem.evaluate_f(x)
        descent = J @ v  # Vector of directional derivatives
        
        while t > 1e-12:
            x_new = self._project(x + t*v)
            f_new = self.problem.evaluate_f(x_new)
            
            # Check Armijo for all objectives
            if all(f_new[i] <= f0[i] + self.beta*t*descent[i] for i in range(len(f0))):
                return t
                
            t /= 2

        print("Line search failed")
        return t

    def _project(self, x):
        """Project variables to feasible bounds"""
        return np.clip(x, self.problem.lb, self.problem.ub)