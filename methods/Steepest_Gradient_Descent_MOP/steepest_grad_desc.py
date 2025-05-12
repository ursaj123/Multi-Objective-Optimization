# File: multi_objective_gradient.py
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
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
            # print(f"Iteration {k+1}/{self.max_iter}")
            # print(f"Current x: {x}")
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

    # def _compute_direction(self, J):
    #     """Solve QP: min α + ½||v||² s.t. Jv ≤ α"""
    #     m, n = J.shape
        
    #     # Solve using scipy's QP solver
    #     def obj(z):
    #         return z[0] + 0.5 * z[1:].dot(z[1:])
            
    #     constraints = [{
    #         'type': 'ineq',
    #         'fun': lambda z, i=i: -J[i].dot(z[1:]) + z[0],
    #         # 'jac': lambda z, i=i: np.concatenate((np.zeros(1), -J[i])),
    #     } for i in range(m)]
        
    #     res = minimize(obj, x0=np.zeros(n+1), 
    #                   constraints=constraints, 
    #                   method='SLSQP',
    #                 #   options={'ftol': 1e-8}
    #     )
        
    #     if not res.success:
    #         # x = x + np.random.uniform(low=-0.1, high=0.1, size=n)
    #         print(f"QP failed: {res.message}")
    #         return np.zeros(n), 0.0
             
            
    #     return res.x[1:], res.x[0]

    def _compute_direction(self, J):
        """Solve QP using trust-constr with explicit constraints"""
        m, n = J.shape
        eps = 1e-10  # Small regularization factor

        # Define optimization variables: z = [α, v_1, ..., v_n]
        dim = n + 1  # α + n variables

        # Objective function: α + ½||v||² + ε(α² + ||v||²)
        def objective(z):
            alpha = z[0]
            v = z[1:]
            return alpha + 0.5 * v.dot(v) + eps * (alpha**2 + v.dot(v))

        # Jacobian of objective
        def jac(z):
            grad = np.zeros_like(z)
            grad[0] = 1 + 2*eps*z[0]
            grad[1:] = z[1:] * (1 + 2*eps)
            return grad

        # Hessian of objective (constant for quadratic terms)
        def hess(z):
            H = np.diag(2*eps * np.ones(dim))
            H[0, 0] = 2*eps  # α² term
            H[1:, 1:] += np.eye(n)  # ||v||² term
            return H

        # Linear constraints: Jv - α ≤ 0 → [-1, J_i]·z ≤ 0
        A = np.hstack([-np.ones((m, 1)), J])  # Constraint matrix
        linear_constraints = LinearConstraint(
            A, 
            lb=-np.inf, 
            ub=0.0
        )

        # Variable bounds (none for α, optional for v)
        bounds = [
            (-np.inf, np.inf)  # α
        ] + [
            (-1, 1) for _ in range(n)  # v
        ]

        # Initial guess
        z0 = np.zeros(dim)

        # Solve with trust-constr
        res = minimize(
            objective,
            z0,
            method='trust-constr',
            jac=jac,
            hess=hess,
            constraints=[linear_constraints],
            bounds=bounds,
            options={'verbose': 0, 'gtol': 1e-9, 'xtol': 1e-9}
        )

        if not res.success:
            return np.zeros(n), 0.0  # Fallback

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