import numpy as np
from numpy.linalg import norm
from time import time
from jaxopt.projection import projection_box
from jaxopt.prox import prox_lasso

from scipy.optimize import (
    BFGS,
    Bounds,
    LinearConstraint,
    OptimizeResult,
    minimize,
    minimize_scalar,
)

class MultiObjectiveADMM:
    def __init__(self, problem, rho=1.0, beta=0.5, epsilon=1e-6, max_outer=100,
                 eps_inner=1e-6, max_inner=50, g_type=('zero', {}), warm_start=False,
                 norm_constraint='Linf', verbose=True):
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
        self.x_history = []
        self.dx_history = []
        self.dz_history = []
        self.lambda_history = []
        self.global_residuals = []

    def initialize_x(self):
        x0 = np.random.uniform(-1.0, 1.0, self.n)
        if self.problem.bounds is not None:
            x0 = np.array([np.random.uniform(lb, ub) for lb, ub in self.problem.bounds])
        z0 = x0.copy()
        return x0, z0

    def x_subproblem(self, x_k, d_z, lambda_):
        grad_f = self.problem.evaluate_gradients_f(x_k)
        
        def objective(params):
            d_x, t = params[:self.n], params[self.n]
            quadratic_term = 0.5 * self.rho * np.linalg.norm(d_x - d_z + lambda_/self.rho)**2
            return t + quadratic_term
        
        cons = [{'type': 'ineq', 
                'fun': lambda params, i=i: params[self.n] - grad_f[i] @ params[:self.n]} 
               for i in range(self.problem.m)]
        
        bounds = [(None, None)]*(self.n + 1)
        if self.norm_constraint == 'Linf':
            bounds = [(-1.0, 1.0) for _ in range(self.n)] + [(None, None)]


        # let's also calculate t value at last history
        t_value = 0.0
        if self.warm_start and self.dx_history:
            t_value = max([(grad_f[i]@self.dx_history[-1]) for i in range(len(grad_f))])
        initial_guess = np.hstack([self.dx_history[-1], t_value]) if (self.warm_start and self.dx_history) else np.zeros(self.n + 1)

        res = minimize(objective, initial_guess, 
                      constraints=cons,
                      bounds=bounds if self.norm_constraint == 'Linf' else None,
                      method='SLSQP')
        
        if not res.success:
            # raise RuntimeError("x-subproblem failed to converge")
            print("x-subproblem failed to converge")
        return res.x[:self.n]

    def prox_wsum_g(self, weights, x, bounds=None):
        """
        solves the problem
            min_{y} (||y||_1 + 0.5*||x-y||_2^2)
        """
        l1_ratios, l1_shifts = self.problem.l1_ratios, self.problem.l1_shifts
        coef = weights * l1_ratios
        x = prox_lasso(
            x + np.sum(coef[1:]) - l1_shifts[0] + l1_shifts[0], coef[0]
        )
        for i in range(1, self.problem.m):
            x = (
                prox_lasso(x - coef[i] - l1_shifts[i], coef[i])
                + l1_shifts[i]
            )

        if bounds is not None:
            x = projection_box(x, (bounds[0], bounds[1]))

        # d_z_p_plus_one = x - z_k
        # return d_z_p_plus_one
        return x


    def z_subproblem(self, z_k, d_x, lambda_):
        # Closed-form solution for special case
        if self.g_type[0] == 'zero':
            # print(f"d_x: {d_x}, z_k: {z_k}, lambda_: {lambda_}")
            if self.norm_constraint == '':
                return d_x + lambda_/self.rho
            else:
                # l_infinity norm constraint
                def objective(d_z):
                    return 0.5 * self.rho * np.linalg.norm(d_x - d_z + lambda_/self.rho)**2
                

                bounds = [(-1.0, 1.0) for _ in range(self.n)]
                initial_guess = self.dz_history[-1] if (self.warm_start and self.dz_history) else np.zeros(self.n)

                res = minimize(objective, initial_guess,
                            bounds=bounds,
                            method='SLSQP')
                
                if not res.success:
                    if self.verbose:
                        # raise RuntimeError("z-subproblem failed to converge")
                        print("z-subproblem failed to converge - g_type = zero, norm_constraint = Linf")                
                return res.x

        elif self.g_type[0] == 'L1':
            optimal_weights = np.ones(self.problem.m)/self.problem.m
            # print(f"d_x: {d_x}, z_k: {z_k}, lambda_: {lambda_}")
            d_x_p_plus_one_plus_u_p_plus_z_k = d_x + lambda_/self.rho + z_k

            def solve_z_dual_subproblem(weights):
                """
                it solves the subproblem:
                    max_{weights} (min_{D_z} (weights@g(D_z) + (rho/2)*||d_x + z_k + lambda_/rho - D_z||^2))
                weights[i]>0 for i=0,1,...,m-1
                sum(weights[i]) = 1
                """

                primal_var_D_z = np.zeros(self.problem.n)
                if self.norm_constraint == 'Linf':
                    inf_norm = np.max(z_k)
                    bounds = (-1-inf_norm, 1+inf_norm)
                    primal_var_D_z = self.prox_wsum_g(weights/self.rho, d_x_p_plus_one_plus_u_p_plus_z_k, bounds)
                else:
                    primal_var_D_z = self.prox_wsum_g(weights/self.rho, d_x_p_plus_one_plus_u_p_plus_z_k)
                # print('Primal var D_z:', primal_var_D_z)

                g_primal = self.problem.evaluate_g(primal_var_D_z)
                fun_val = weights@g_primal + (self.rho/2)*norm(primal_var_D_z - d_x_p_plus_one_plus_u_p_plus_z_k)**2
                jac = g_primal

                return -fun_val, -jac

            

            if self.problem.m==2:
                res = minimize_scalar(lambda w: solve_z_dual_subproblem(np.array([w, 1-w]))[0], bounds=(0, 1))
                if not res.success:
                    if self.verbose:
                        # raise RuntimeError("z-subproblem failed to converge")
                        print("z-subproblem failed to converge - m=2, case g_type = L1")
                
                optimal_weights = np.array([res.x, 1-res.x])
            elif self.problem.m>2:
                # for m>2, we can use the same method as above
                res = minimize(
                    fun=solve_z_dual_subproblem,
                    x0=np.ones(self.problem.m)/self.problem.m,
                    method="trust-constr",
                    jac=True,
                    hess=BFGS(),
                    bounds=Bounds(lb=0, ub=np.inf),
                    constraints=LinearConstraint(np.ones(self.problem.m), lb=1, ub=1)
                )
                if not res.success:
                    if self.verbose:
                        # raise RuntimeError("z-subproblem failed to converge")
                        print("z-subproblem failed to converge - m>2, case g_type = L1")
                
                optimal_weights = res.x
            
            # now that we ave optimal weights, we can calculate the primal variable D_z by the prox operator
            if self.norm_constraint == 'Linf':
                inf_norm = np.max(z_k)
                bounds = (-1-inf_norm, 1+inf_norm)
                primal_var_D_z = self.prox_wsum_g(optimal_weights/self.rho, d_x_p_plus_one_plus_u_p_plus_z_k, bounds)
            else:
                primal_var_D_z = self.prox_wsum_g(optimal_weights/self.rho, d_x_p_plus_one_plus_u_p_plus_z_k)
            
            return primal_var_D_z - z_k



    def subadmm(self, x_k, z_k):
        # Initialize with warm start values if available
        lambda_p = self.lambda_history[-1] if (self.warm_start and self.lambda_history) else np.zeros(self.n)
        d_z_p = self.dz_history[-1] if (self.warm_start and self.dz_history) else np.zeros(self.n)
        
        for p in range(self.max_inner):
            # print(f"SubADMM inner iteration {p+1}/{self.max_inner}")

            # print(f"Solving x-subproblem at inner iteration {p+1}")
            d_x_new = self.x_subproblem(x_k, d_z_p, lambda_p)
            # print(f"Solving z-subproblem at inner iteration {p+1}")
            d_z_new = self.z_subproblem(z_k, d_x_new, lambda_p)
            lambda_new = lambda_p + self.rho * (d_x_new - d_z_new)
            
            # SubADMM residuals
            primal_res = np.linalg.norm(d_x_new - d_z_new)
            dual_res = np.linalg.norm(d_z_new - d_z_p)
            # print(f"inner iteration: {p+1}\nlambda_new: {lambda_new} d_x_new: {d_x_new}, d_z = {d_z_p}, d_z_new: {d_z_new}, Primal residual: {primal_res}, Dual residual: {dual_res}")
            
            # Update for next iteration
            d_z_p, lambda_p = d_z_new, lambda_new

            # Check convergence
            if primal_res < self.eps_inner and dual_res < self.eps_inner:
                if self.verbose:
                    print(f"SubADMM converged at inner iteration {p}")
                break

        return d_x_new, d_z_new, lambda_new

    def project(self, x):
        """
        Projects x onto the feasible set defined by the problem bounds.
        If no bounds are defined, it returns x unchanged.
        """
        if self.problem.bounds is None:
            return x
        else:
            lbs, ubs = [], []
            for lb, ub in self.problem.bounds:
                lbs.append(lb)
                ubs.append(ub)
            # lbs, ubs = zip(*self.problem.bounds)
            lbs, ubs = np.array(lbs), np.array(ubs)
            return np.clip(x, a_min=lbs, a_max=ubs)

    def line_search(self, x_k, z_k, d_x, d_z):
        t = 1.0
        f_curr = self.problem.evaluate_f(x_k)
        g_curr = self.problem.evaluate_g(z_k)
        grad_f = self.problem.evaluate_gradients_f(x_k)
        descent = np.array([grad @ d_x for grad in grad_f])

        bnds = self.problem.bounds
        lbs, ubs = [], []
        if bnds is not None:
            for lb, ub in bnds:
                lbs.append(lb)
                ubs.append(ub)
            lbs = np.array(lbs)
            ubs = np.array(ubs)
        
        while t > 1e-10:
            x_new = x_k + t*d_x
            z_new = z_k + t*d_z
            f_new = self.problem.evaluate_f(x_new)
            g_new = self.problem.evaluate_g(z_new)

            # gotta check the bounds also
            
            bounds_check_x = np.all(x_new >= lbs) and np.all(x_new <= ubs)
            bounds_check_z = np.all(z_new >= lbs) and np.all(z_new <= ubs)
            val_check = np.all(f_new + g_new <= f_curr + g_curr + self.beta * t * descent)
            # bounds_check = True
            if val_check and bounds_check_x and bounds_check_z:
                return t
            t *= 0.5
        
        if t<=1e-10:
            t = 0.0

        print("Line search failed to converge")
        return t

    def solve(self, x, z):
        start_time = time()
        comp_time = None

        for k in range(self.max_outer):
            # print(f"k = {k}, x = {x}")
            # print(f"Outer iteration {k+1}/{self.max_outer}")
            # Solve subADMM for directions
            # print(f"SubADMM started at outer iteration {k+1}")
            d_x, d_z, lambda_new = self.subadmm(x, z)
            # print(f"SubADMM finished at outer iteration {k+1}")
            # print(f"Direction x: {d_x}, Direction z: {d_z}")
            
            # Line search
            t = self.line_search(x, z, d_x, d_z)
            # t = 1
            # print(f"k = {k} t = {t:.4f}, d_x: {d_x}, d_z: {d_z}, lambda_new: {lambda_new}")
            
            # Update primal variables
            
            x_new = x + t*d_x
            z_new = z + t*d_z
            # x_new = self.project(x_new)
            # z_new = self.project(z_new)
            
            # Global convergence check (using x difference)
            global_res = np.sqrt(np.linalg.norm(x_new - x)**2 + np.linalg.norm(z_new - z)**2)
            # global_res = np.sqrt(np.linalg.norm(d_x)**2 + np.linalg.norm(d_z)**2)
            # self.global_residuals.append(global_res)
            # if self.verbose:
            #     print(f"Iter {k}: Global residual={global_res:.2e}")
                
            # Update history
            x, z = x_new, z_new
            # self.x_history.append(x.copy())
            # self.dx_history.append(d_x)
            # self.dz_history.append(d_z)
            # self.lambda_history.append(lambda_new)

            # print(f"x_new: {x_new}, z_new: {z_new}, t: {t}, global_res: {global_res}")

            if global_res < self.epsilon:
                print(f"Global convergence at iteration {k+1}")
                comp_time = time() - start_time
                break

        comp_time = time() - start_time
        x = self.project(x)
        z = self.project(z)

            
        return {'x':x, 'z':z, 'iterations':k,
        'f':self.problem.evaluate(x, z), 
        'residual':global_res, 
        'runtime':comp_time}