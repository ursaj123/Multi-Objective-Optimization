import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, Bounds
import time

class MultiObjectiveNewton:
    """
    Implements Newton's method for unconstrained multiobjective optimization
    as described in the paper "Newton's Method for Multiobjective Optimization"
    by J. Fliege, L. M. Graña Drummond, and B. F. Svaiter (SIAM J. OPTIM., 2009).
    """
    def __init__(self, problem, sigma=0.1, max_iters=100, tol_theta=1e-6, verbose=False, subproblem_solver='trust-constr'):
        self.problem = problem
        self.sigma = sigma
        self.max_iters = max_iters
        self.tol_theta = tol_theta
        self.verbose = verbose
        self.subproblem_solver = subproblem_solver

        if not (0 < self.sigma < 1):
            raise ValueError("Sigma for line search must be between 0 and 1.")
        if self.subproblem_solver not in ['trust-constr', 'SLSQP']:
            raise ValueError("subproblem_solver must be 'trust-constr' or 'SLSQP'.")


    def _solve_direction_subproblem(self, x_k):
        grads_fk = self.problem.evaluate_gradients_f(x_k)
        hesss_fk = self.problem.evaluate_hessians_f(x_k)

        m = self.problem.m
        n = self.problem.n
        initial_guess = np.zeros(n + 1)

        def objective_fn(variables):
            return variables[n] 

        constraints = []
        if self.subproblem_solver == 'trust-constr':
            for j_idx in range(m):
                grad_j = grads_fk[j_idx].flatten()
                hess_j = hesss_fk[j_idx]
                def con_fun_j(variables, grad_val=grad_j, hess_val=hess_j):
                    s_vec = variables[:n]
                    t_scalar = variables[n]
                    return grad_val.T @ s_vec + 0.5 * s_vec.T @ hess_val @ s_vec - t_scalar
                def con_jac_j(variables, grad_val=grad_j, hess_val=hess_j):
                    s_vec = variables[:n]
                    jac_s = grad_val + hess_val @ s_vec
                    return np.concatenate((jac_s, [-1.0]))
                def con_hess_j(variables, p_vec, grad_val=grad_j, hess_val=hess_j):
                    hess_prod_s = hess_val @ p_vec[:n]
                    return np.concatenate((hess_prod_s, [0.0]))
                nlc = NonlinearConstraint(con_fun_j, -np.inf, 0, jac=con_jac_j, hess=con_hess_j)
                constraints.append(nlc)
        
        elif self.subproblem_solver == 'SLSQP':
            for j_idx in range(m):
                grad_val = grads_fk[j_idx].flatten()
                hess_val = hesss_fk[j_idx]
                def con_fun_slsqp(vars_in, grad=grad_val, hess=hess_val):
                    s_in = vars_in[:n]
                    t_in = vars_in[n]
                    val = grad.T @ s_in + 0.5 * s_in.T @ hess @ s_in
                    return t_in - val 
                constraints.append({'type': 'ineq', 'fun': con_fun_slsqp})
        
        sub_problem_bounds = None 

        try:
            res = minimize(objective_fn, initial_guess, method=self.subproblem_solver,
                           jac='2-point', hess=None, constraints=constraints, bounds=sub_problem_bounds,
                           options={'disp': (self.verbose > 1), 
                                    'gtol': 1e-8 if self.subproblem_solver == 'trust-constr' else 1e-9, # trust-constr uses gtol, SLSQP ftol
                                    'xtol': 1e-8 if self.subproblem_solver == 'trust-constr' else 1e-9,
                                    'ftol': 1e-9 if self.subproblem_solver == 'SLSQP' else None, # SLSQP specific
                                    'verbose': (self.verbose > 2 if self.subproblem_solver=='trust-constr' else 0) })

            if res.success or (self.subproblem_solver == 'trust-constr' and res.status in [1,2,3]):
                s_k = res.x[:n]
                theta_k = res.fun 
                return s_k, theta_k, True
            else:
                if self.verbose: print(f"Warning: Subproblem solver ({self.subproblem_solver}) failed. Msg: {res.message}. Status: {res.status}")
                return np.zeros(n), 0.0, False
        except Exception as e:
            if self.verbose: print(f"Exception in subproblem solver ({self.subproblem_solver}): {e}")
            return np.zeros(n), 0.0, False

    def _line_search(self, x_k, f_k, s_k, theta_k):
        t_step = 1.0
        m = self.problem.m
        problem_bnds = getattr(self.problem, 'problem_bounds', None)

        for _ in range(20): 
            x_candidate = x_k + t_step * s_k
            valid_candidate = True
            if problem_bnds:
                for var_idx in range(self.problem.n):
                    low, high = problem_bnds[var_idx]
                    if not ( (low is None or x_candidate[var_idx] >= low - 1e-9) and \
                             (high is None or x_candidate[var_idx] <= high + 1e-9) ):
                        valid_candidate = False; break
            if not valid_candidate:
                if self.verbose > 1: print(f"LS: t={t_step:.2e}, x_cand {x_candidate} out of bounds {problem_bnds}")
                t_step /= 2; continue
            try:
                f_candidate = self.problem.evaluate_f(x_candidate)
            except Exception as e:
                 if self.verbose > 1: print(f"LS: t={t_step:.2e}, eval F at x_cand failed: {e}")
                 t_step /= 2; continue

            armijo_satisfied_all_j = True
            for j_obj in range(m):
                rhs_armijo = f_k[j_obj] + self.sigma * t_step * theta_k
                if f_candidate[j_obj] > rhs_armijo + 1e-9 * abs(f_k[j_obj] + 1e-9): 
                    armijo_satisfied_all_j = False; break
            if armijo_satisfied_all_j:
                if self.verbose > 1: print(f"LS: Accepted t={t_step:.2e}")
                return t_step
            
            j_obj_local = locals().get('j_obj', -1) 
            if self.verbose > 1:
                 print(f"LS: t={t_step:.2e} rejected. F_cand[{j_obj_local}]={(f_candidate[j_obj_local] if j_obj_local !=-1 and j_obj_local < len(f_candidate) else 'N/A')} vs RHS (th={theta_k:.2e})")
            t_step /= 2
        if self.verbose: print("LS failed after max backtracking.")
        return 0.0

    def solve(self, x0):
        start_time = time.time()
        x_k = np.array(x0, dtype=float)
        if x_k.ndim == 0: x_k = np.array([x_k])

        converged = False
        final_x = x_k
        final_f = None
        iterations_done = 0

        try:
            current_f_k = self.problem.evaluate_f(x_k)
        except Exception as e:
            runtime = time.time() - start_time
            if self.verbose: print(f"Error evaluating F at initial x0: {e}")
            return {'x': final_x, 'f': None, 'runtime': runtime, 'converged': False, 'iterations': 0, 'message': f"Error at initial eval: {e}"}
        
        if self.verbose: print(f"Starting M-O Newton ({self.subproblem_solver}). Initial x0: {x_k}, F(x0): {current_f_k}")

        for k_iter in range(self.max_iters):
            iterations_done = k_iter + 1
            final_x = x_k # Keep track of last valid x
            final_f = current_f_k

            s_k, theta_k, sub_success = self._solve_direction_subproblem(x_k)
            
            if not sub_success and self.verbose: print(f"Iter {k_iter+1}: Subproblem issues.")
            
            t_k = self._line_search(x_k, current_f_k, s_k, theta_k)
            if t_k <= 1e-12:
                converged = False # Did not converge if line search fails
                if self.verbose: print(f"Iter {k_iter+1}: Line search failed (t_k={t_k:.2e}). Stopping.")
                runtime = time.time() - start_time
                return {'x': final_x, 'f': final_f, 'runtime': runtime, 'converged': converged, 'iterations': iterations_done, 'message': "Stopped: Line search failed/small step."}

            x_k = x_k + t_k * s_k 
            current_f_k = self.problem.evaluate_f(x_k)
            if self.verbose: print(f"Iter {k_iter+1}: th={theta_k:.2e}, s={s_k}, t={t_k:.2e}, x={x_k}, F(x)={current_f_k}")

            if np.abs(theta_k) <= self.tol_theta:
                converged = True
                msg = "Converged: theta_k approx 0."
                if self.verbose: print(f"Iter {k_iter+1}: Stopping. theta={theta_k:.2e} >= {-self.tol_theta:.1e}.")
                if np.abs(theta_k) > 1e-7 and sub_success: msg = f"Warning: theta_k={theta_k:.2e} > 0."
                elif not sub_success: msg = "Stopped: Subproblem failed & theta_k approx 0."
                runtime = time.time() - start_time
                return {'x': x_k, 'f': current_f_k, 'runtime': runtime, 'converged': converged, 'iterations': iterations_done, 'message': msg}

        # If loop finishes, max_iters reached
        converged = False 
        final_x = x_k
        final_f = current_f_k
        runtime = time.time() - start_time
        if self.verbose: print("Reached max iterations.")
        return {'x': final_x, 'f': final_f, 'runtime': runtime, 'converged': converged, 'iterations': self.max_iters, 'message': "Max iterations reached."}

# if __name__ == '__main__':
#     print("--- Testing MultiObjectiveNewton with AP2 Problem ---")
#     problem_ap2 = AP2(lb=-5.0, ub=5.0)

#     test_cases = {
#         "x0 = -2.0": np.array([-2.0]),
#         "x0 = 0.5 (Pareto optimal)": np.array([0.5]),
#         "x0 = 3.0": np.array([3.0]),
#         "x0 = 0.0 (Pareto boundary)": np.array([0.0]),
#         "x0 = 1.0 (Pareto boundary)": np.array([1.0]),
#     }
    
#     solvers_to_test = ['trust-constr', 'SLSQP'] 

#     for solver_name in solvers_to_test:
#         print(f"\n\n--- TESTING WITH SUBPROBLEM SOLVER: {solver_name} ---")
#         for name, x0_val in test_cases.items():
#             print(f"\n--- Test Case: {name} (using {solver_name}) ---")
#             newton_solver = MultiObjectiveNewton(problem_ap2, sigma=0.1, max_iters=50, 
#                                                  tol_theta=1e-7, verbose=True, subproblem_solver=solver_name)
#             result_obj = newton_solver.solve(x0=x0_val)
            
#             print("\nReturned Object:")
#             print(f"  x: {result_obj['x']}")
#             print(f"  f: {result_obj['f']}")
#             print(f"  runtime: {result_obj['runtime']:.4f}s")
#             print(f"  converged: {result_obj['converged']}")
#             print(f"  iterations: {result_obj['iterations']}")
#             if 'message' in result_obj:
#                 print(f"  message: {result_obj['message']}")