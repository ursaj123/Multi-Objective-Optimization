import numpy as np
from scipy.optimize import minimize
from scipy.linalg import norm

class MultiObjectiveProblem:
    def __init__(self, f_list, grad_f_list, bounds, g_types):
        self.f_list = f_list              # List of objective functions
        self.grad_f_list = grad_f_list    # List of gradient functions
        self.bounds = bounds              # Bounds: [[a1, b1], ..., [an, bn]]
        self.g_types = g_types            # List of strings: ['zero', 'l1', ...]

def initialize_x(bounds):
    """Initialize x within bounds"""
    return np.array([np.random.uniform(low, high) for (low, high) in bounds])

def x_subproblem_multi(x, grad_list, rho, d_z, lam_matrix):
    """Solve x-direction subproblem for multiple lambdas: min max_i [∇f_i^T d_x + (ρ/2)||d_x - d_z + λ_i/ρ||^2]"""
    n = len(x)
    m = len(grad_list)
    
    def objective(vars):
        d_x, alpha = vars[:n], vars[n]
        return alpha + sum((rho/2) * norm(d_x - d_z + lam_matrix[i]/rho)**2 for i in range(m))
    
    constraints = [
        {'type': 'ineq', 'fun': lambda vars, i=i: vars[n] - np.dot(grad_list[i], vars[:n])}
        for i in range(m)
    ] + [{'type': 'ineq', 'fun': lambda vars: 1 - norm(vars[:n])}]
    
    res = minimize(objective, x0=np.zeros(n+1), constraints=constraints, method='SLSQP')
    return res.x[:n], res.x[n]

def z_subproblem_multi(d_x, lam_matrix, rho, g_types):
    """Solve z-direction subproblem for multiple g_types (only 'zero' implemented)"""
    n = len(d_x)
    m = lam_matrix.shape[0]
    
    if not all(g == 'zero' for g in g_types):
        raise NotImplementedError("Mixed g_types not supported")
    
    # For g=0, solve: min ∑_i (ρ/2)||d_x - d_z + λ_i/ρ||^2 with ||d_z|| <= 1
    def objective(d_z):
        return sum((rho/2) * norm(d_x - d_z + lam_matrix[i]/rho)**2 for i in range(m))
    
    cons = {'type': 'ineq', 'fun': lambda d_z: 1 - norm(d_z)}
    res = minimize(objective, x0=np.zeros(n), constraints=cons, method='SLSQP')
    return res.x

def subadmm_multi_lambda(x, grad_list, g_types, rho, eps_inner, max_inner):
    """SubADMM inner loop for Algorithm 2b"""
    n = len(x)
    m = len(grad_list)
    lam_matrix = np.zeros((m, n))  # Dual variables (unscaled)
    d_z = np.zeros(n)
    
    for p in range(max_inner):
        # X-direction update
        d_x, alpha = x_subproblem_multi(x, grad_list, rho, d_z, lam_matrix)
        
        # Z-direction update (assume all g_i are 'zero')
        d_z_new = z_subproblem_multi(d_x, lam_matrix, rho, g_types)
        
        # Dual update: λ_i = λ_i + ρ(d_x - d_z)
        lam_matrix_new = lam_matrix + rho * (d_x - d_z_new)
        
        # Termination check
        primal_res = norm(d_x - d_z_new)
        dual_res = norm(lam_matrix_new - lam_matrix)
        if primal_res < eps_inner and dual_res < eps_inner:
            break
        d_z, lam_matrix = d_z_new, lam_matrix_new
    
    return d_x, d_z

def algo_2b(mop, rho=1.0, beta=0.5, epsilon=1e-6, max_outer=100, eps_inner=1e-6, max_inner=10):
    """Main algorithm with multiple lambdas"""
    n = len(mop.bounds)
    x = initialize_x(mop.bounds)
    x_history = [x.copy()]
    
    for _ in range(max_outer):
        # Compute gradients at current x
        grad_list = [grad_f(x) for grad_f in mop.grad_f_list]
        
        # Solve SUBADMM for directions
        d_x, d_z = subadmm_multi_lambda(x, grad_list, mop.g_types, rho, eps_inner, max_inner)
        
        # Line search with bounds check
        t = 1.0
        for _ in range(10):
            x_new = x + t * d_x
            # Check if within bounds
            in_bounds = all(low <= xi <= high for xi, (low, high) in zip(x_new, mop.bounds))
            if in_bounds:
                # Check Armijo condition for all objectives
                armijo_ok = all(
                    mop.f_list[i](x_new) <= mop.f_list[i](x) + beta * t * np.dot(grad_list[i], d_x)
                    for i in range(len(mop.f_list))
                )
                if armijo_ok:
                    break
            t *= 0.5
        
        # Update x and check termination
        x_prev = x.copy()
        x = x_new.copy()
        x_history.append(x.copy())
        if norm(x - x_prev) < epsilon:
            break
    
    return np.array(x_history)