import numpy as np
from scipy.optimize import minimize

class MultiObjectiveADMM:
    def __init__(self, problem, rho=1.0, beta=0.5, epsilon=1e-6, max_outer=100,
                 eps_inner=1e-6, max_inner=50, g_type=('zero', {}), warm_start=False,
                 norm_constraint='L2', verbose=True):
        self.problem = problem
        self.rho = rho
        self.beta = beta
        self.epsilon = epsilon
        self.max_outer = max_outer
        self.eps_inner = eps_inner
        self.max_inner = max_inner
        self.g_type = g_type
        self.warm_start = warm_start
        self.norm_constraint = norm_constraint
        self.verbose = verbose
        self.n = problem.n
        self.dx_history = []
        self.dz_history = []
        
    def initialize_x(self):
        # Initialize x and z within bounds uniformly
        x0 = np.array([np.random.uniform(self.problem.lb[i], self.problem.ub[i]) for i in range(self.n)])
        z0 = x0.copy()
        return x0, z0
    
    def x_subproblem(self, x_k, d_x_prev, lambda_prev):
        grad_f = self.problem.evaluate_gradients_f(x_k)
        
        def objective(params):
            d_x = params[:self.n]
            t = params[self.n]
            quad_term = 0.5 * self.rho * np.linalg.norm(d_x - d_x_prev + lambda_prev)**2
            return t + quad_term
        
        cons = []
        # Max gradient constraints: grad_f_i^T d_x <= t for all i
        for i in range(self.problem.m):
            cons.append({'type': 'ineq',
                         'fun': lambda params, i=i: params[self.n] - grad_f[i] @ params[:self.n]})
        
        # Norm constraint
        if self.norm_constraint == 'L2':
            cons.append({'type': 'ineq',
                         'fun': lambda params: 1 - np.linalg.norm(params[:self.n])})
        elif self.norm_constraint == 'Linf':
            cons.append({'type': 'ineq',
                         'fun': lambda params: 1 - np.max(np.abs(params[:self.n]))})
        
        # Initial guess (warm start or zeros)
        if self.warm_start and self.dx_history:
            initial_guess = np.hstack([self.dx_history[-1], 0.0])
        else:
            initial_guess = np.zeros(self.n + 1)
        
        res = minimize(objective, initial_guess, constraints=cons, method='SLSQP')
        if not res.success:
            raise ValueError("x_subproblem failed to converge")
        d_x = res.x[:self.n]
        return d_x
    
    def z_subproblem(self, z_k, d_z_prev, lambda_prev):
        # For g_type='zero', the max term vanishes
        def objective(d_z):
            return 0.5 * self.rho * np.linalg.norm(d_z - d_z_prev + lambda_prev)**2
        
        cons = []
        if self.norm_constraint == 'L2':
            cons.append({'type': 'ineq', 'fun': lambda d_z: 1 - np.linalg.norm(d_z)})
        elif self.norm_constraint == 'Linf':
            cons.append({'type': 'ineq', 'fun': lambda d_z: 1 - np.max(np.abs(d_z))})
        
        # Initial guess
        if self.warm_start and self.dz_history:
            initial_guess = self.dz_history[-1]
        else:
            initial_guess = np.zeros(self.n)
        
        res = minimize(objective, initial_guess, constraints=cons, method='SLSQP')
        if not res.success:
            raise ValueError("z_subproblem failed to converge")
        return res.x
    
    def subadmm(self, x_k, z_k):
        lambda_p = np.zeros(self.n)
        d_x_p = np.zeros(self.n)
        d_z_p = np.zeros(self.n)
        
        for p in range(self.max_inner):
            # x-direction update
            d_x_new = self.x_subproblem(x_k, d_x_p, lambda_p)
            
            # z-direction update
            d_z_new = self.z_subproblem(z_k, d_z_p, lambda_p)
            
            # Dual update
            lambda_new = lambda_p + self.rho * (d_x_new - d_z_new)
            
            # Termination check
            primal_res = np.linalg.norm(d_x_new - d_z_new)
            dual_res = np.linalg.norm(d_z_new - d_z_p)
            if primal_res <= self.eps_inner and dual_res <= self.eps_inner:
                if self.verbose:
                    print(f"SubADMM converged at inner iteration {p}, primal residual: {primal_res}, dual residual: {dual_res}")
                break
                
            # Update for next iteration
            d_x_p, d_z_p, lambda_p = d_x_new, d_z_new, lambda_new
        
        return d_x_p, d_z_p
    
    def line_search(self, x_k, z_k, d_x, d_z):
        t = 1.0
        f_current = self.problem.evaluate_f(x_k)
        g_current = self.problem.evaluate_g(z_k)
        grad_f = self.problem.evaluate_gradients_f(x_k)
        descent = np.array([grad @ d_x for grad in grad_f])
        
        while t > 1e-10:
            x_new = x_k + t * d_x
            z_new = z_k + t * d_z
            f_new = self.problem.evaluate_f(x_new)
            g_new = self.problem.evaluate_g(z_new)
            # will check this also at some time
            # if np.all(f_new  + g_new <= f_current + g_current + self.beta * t * descent):
            if np.all(f_new <= f_current + self.beta * t * descent)
                return t
            t *= 0.5
        
        if self.verbose:
            print("Line search failed to find a suitable step size")
        return t
    
    def solve(self):
        x, z = self.initialize_x()
        for k in range(self.max_outer):
            # SubADMM to get directions
            d_x, d_z = self.subadmm(x, z)
            
            # Line search
            t = self.line_search(x, z, d_x, d_z)
            
            # Primal update
            x_new = x + t * d_x
            z_new = z + t * d_z
            
            # Termination check
            if np.linalg.norm(x_new - x) <= self.epsilon:
                if self.verbose:
                    print(f"Global problem converged at outer iteration {k}")
                break
                
            x, z = x_new, z_new
            self.dx_history.append(d_x)
            self.dz_history.append(d_z)
            
        return x, z