import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Attempt to import pymoo for reference directions
try:
    from pymoo.util.ref_dirs import get_reference_directions
    pymoo_available = True
except ImportError:
    pymoo_available = False
    print("Warning: pymoo library not found. Using simple random reference directions.")
    print("For better reference vector generation, please install pymoo: pip install pymoo")

# --- Problem Definition (Example) ---
def example_objective_functions(x_vars):
    """
    Example bi-objective function.
    x_vars: numpy array of decision variables [x0, x1]
    Returns: numpy array of objective values [f1, f2]
    """
    if x_vars.ndim > 1: # Ensure x_vars is 1D for consistent indexing
        x_vars = x_vars.flatten()
    if x_vars.shape!= (2,):
        raise ValueError(f"x_vars must be a 2-element array, got shape {x_vars.shape}")
        
    f1 = x_vars**2 + x_vars[1]**2
    f2 = (x_vars - 2)**2 + (x_vars[1] - 2)**2
    return np.array([f1, f2])

# --- Achievement Scalarizing Function (ASF) ---
def achievement_scalarizing_function(obj_values, aspiration_point, weights, alpha_asf):
    """
    Calculates the ASF value.
    obj_values: array of m objective function values for a given x_p
    aspiration_point: array of m aspiration levels (reference point r_p or z)
    weights: array of m weights w_p,j
    alpha_asf: augmentation coefficient
    """
    weighted_diff = weights * (obj_values - aspiration_point)
    term1 = np.max(weighted_diff)
    term2 = alpha_asf * np.sum(weighted_diff)
    return term1 + term2

# --- Objective function for x_p-update ---
def x_p_update_objective(x_vars_p, # Current x_p being optimized
                         F_func, # The multi-objective function F(x)
                         aspiration_point_p, # Aspiration levels for this x_p
                         weights_asf_p, # ASF weights for this x_p
                         alpha_asf, rho,
                         y_k_p, u_k_p): # From previous ADMM iteration
    """
    Objective function for the x_p-update step in SO-MO-ADMM.
    Minimizes ASF + proximal term.
    """
    obj_vals = F_func(x_vars_p)
    asf_val = achievement_scalarizing_function(obj_vals, aspiration_point_p, weights_asf_p, alpha_asf)
    prox_term = (rho / 2.0) * np.sum((x_vars_p - y_k_p + u_k_p)**2)
    return asf_val + prox_term

# --- SO-MO-ADMM Algorithm Class ---
class SOMO_ADMM:
    def __init__(self, F_func, P, dim_x, num_obj,
                 rho=1.0, beta=0.1, alpha_asf=0.01,
                 max_iters=100, tol=1e-4,
                 aspiration_scaling_factor=2.0,
                 initial_x_range=(0.0, 2.0)): # Corrected default tuple
        self.F_func = F_func
        self.P_initial_requested = P 
        self.P = P # Current number of candidates, might be adjusted by _initialize_aspiration_levels
        self.dim_x = dim_x
        self.num_obj = num_obj
        self.rho = rho
        self.beta = beta # Weight for the diversity term
        self.alpha_asf = alpha_asf
        self.max_iters = max_iters
        self.tol = tol
        self.aspiration_scaling_factor = aspiration_scaling_factor
        self.initial_x_range = initial_x_range

        # Step 1: Initialize aspiration levels. This might adjust self.P.
        self._initialize_aspiration_levels()

        # Step 2: Initialize P-dependent attributes using the (potentially adjusted) self.P.
        self._initialize_p_dependent_attributes()

    def _initialize_p_dependent_attributes(self):
        """Initializes or re-initializes attributes that depend on the final self.P value."""
        if self.P > 0:
            self.weights_asf = np.ones((self.P, self.num_obj))
            # Corrected np.random.uniform usage
            self.x_vars = np.random.uniform(self.initial_x_range, self.initial_x_range[1],
                                            size=(self.P, self.dim_x))
            self.y_vars = np.copy(self.x_vars)
            self.u_vars = np.zeros((self.P, self.dim_x))

            # Matrix for y_update, assuming diversity term D = - sum_{s<t} ||y_s - y_t||^2
            if self.P > 1:
                diag_val = self.rho - 2 * self.beta * (self.P - 1)
                off_diag_val = 2 * self.beta
                self.A_y_update_matrix = np.full((self.P, self.P), off_diag_val)
                np.fill_diagonal(self.A_y_update_matrix, diag_val)
                try:
                    self.A_inv_y_update = np.linalg.inv(self.A_y_update_matrix)
                except np.linalg.LinAlgError:
                    print("Warning: y-update matrix is singular or near-singular. Using least squares for y-updates.")
                    self.A_inv_y_update = None
            elif self.P == 1: # No diversity term needed, or beta should be 0
                self.A_y_update_matrix = np.array([[self.rho]])
                self.A_inv_y_update = np.array([[1.0/self.rho]]) if self.rho!= 0 else None
            
        else: # self.P == 0
            self.weights_asf = np.array()
            self.x_vars = np.array()
            self.y_vars = np.array()
            self.u_vars = np.array()
            self.A_y_update_matrix = np.array()
            self.A_inv_y_update = None

    def _initialize_aspiration_levels(self):
        """
        Initializes aspiration levels (reference vectors).
        This method may adjust self.P based on the number of reference vectors actually generated.
        """
        P_requested = self.P_initial_requested
        
        if P_requested == 0:
            self.aspiration_levels = np.array()
            self.P = 0 # Final P is 0
            return

        if self.num_obj == 1: # Simple case for single objective
            self.aspiration_levels = np.linspace(0, -0.1 * (P_requested - 1), P_requested).reshape(-1, 1)
            self.P = P_requested # P is as requested
            return

        ref_dirs_candidate = None
        actual_P_generated = 0

        if pymoo_available:
            try: # Try 'energy' method first for its flexibility
                ref_dirs_candidate = get_reference_directions(name="energy", n_dim=self.num_obj, n_points=P_requested, seed=1)
                if ref_dirs_candidate is not None and ref_dirs_candidate.shape > 0:
                    actual_P_generated = ref_dirs_candidate.shape # Correctly get number of points
            except Exception as e_energy:
                print(f"Pymoo 'energy' method failed: {e_energy}. Trying 'das-dennis'.")
                ref_dirs_candidate = None

            if ref_dirs_candidate is None or actual_P_generated == 0: # Fallback to 'das-dennis'
                try:
                    # Approximate n_partitions for Das-Dennis to get around P_requested points
                    if self.num_obj == 2: n_partitions_dd = max(0, P_requested - 1)
                    elif self.num_obj == 3: # N = (H+2)(H+1)/2. Solve for H.
                        H_approx = int(np.ceil((-3 + np.sqrt(1 + 8 * P_requested)) / 2.0))
                        n_partitions_dd = max(1, H_approx) # Ensure at least 1 partition if P_requested > 0
                    else: # For M > 3, estimation is harder, use a heuristic
                        n_partitions_dd = max(1, int(np.ceil(P_requested**(1.0/(self.num_obj-1))))) if P_requested > 0 else 0
                    
                    print(f"Using Das-Dennis with n_partitions={n_partitions_dd} for num_obj={self.num_obj} to aim for P={P_requested}")
                    ref_dirs_candidate = get_reference_directions("das-dennis", self.num_obj, n_partitions=n_partitions_dd)
                    if ref_dirs_candidate is not None and ref_dirs_candidate.shape > 0:
                        actual_P_generated = ref_dirs_candidate.shape # Correctly get number of points
                except Exception as e_dd:
                    print(f"Pymoo 'das-dennis' method failed: {e_dd}. Using random normalized directions.")
                    ref_dirs_candidate = None
                    actual_P_generated = 0
        
        # Final decision on reference directions and P
        if ref_dirs_candidate is not None and actual_P_generated > 0:
            if actual_P_generated!= P_requested:
                print(f"Pymoo generated {actual_P_generated} reference directions, while {P_requested} were requested.")
                # Option: Adjust self.P to actual_P_generated
                self.P = actual_P_generated 
                print(f"Number of candidates P has been adjusted to {self.P}.")
                # If actual_P_generated > P_requested, we could take a subset
                if actual_P_generated > P_requested:
                     indices = np.random.choice(actual_P_generated, P_requested, replace=False)
                     ref_dirs_candidate = ref_dirs_candidate[indices,:]
                     self.P = P_requested
                     print(f"Selected {self.P} candidates from the {actual_P_generated} generated.")
                # If actual_P_generated < P_requested, we use what we have.
            else:
                self.P = P_requested # Pymoo gave exactly what was asked

            final_ref_dirs = ref_dirs_candidate
        else: # Fallback to random if pymoo failed or returned nothing
            print(f"Warning: Could not generate reference directions using pymoo. Using {P_requested} random normalized directions.")
            final_ref_dirs = np.random.rand(P_requested, self.num_obj) + 1e-9 # Add small epsilon for normalization
            final_ref_dirs = final_ref_dirs / np.sum(final_ref_dirs, axis=1, keepdims=True)
            self.P = P_requested # Use the requested P

        if self.P == 0 or final_ref_dirs is None or final_ref_dirs.shape == 0:
            print("Error: No reference directions could be generated or P is 0. Algorithm cannot proceed.")
            self.aspiration_levels = np.array()
            self.P = 0 # Ensure P is 0 if no refs
            return

        self.aspiration_levels = final_ref_dirs * self.aspiration_scaling_factor

    def solve(self):
        history = {'primal_res':[], 'dual_res':[]}

        if self.P == 0:
            print("Number of Pareto candidates P is 0. Nothing to solve.")
            return np.array(), np.array(), history

        for k_iter in range(self.max_iters):
            y_vars_old_for_dual_res = np.copy(self.y_vars)

            # 1. x_p-updates (parallelizable)
            for p_idx in range(self.P):
                initial_guess_x_p = self.x_vars[p_idx, :]
                # For constrained problems, bounds could be set here:
                # bounds_x_p = [(min_val, max_val)] * self.dim_x 
                bounds_x_p = None # Unconstrained for this example

                obj_func_for_scipy = lambda x_v_p: x_p_update_objective(
                    x_v_p, self.F_func,
                    self.aspiration_levels[p_idx, :], self.weights_asf[p_idx, :],
                    self.alpha_asf, self.rho,
                    self.y_vars[p_idx, :], self.u_vars[p_idx, :]
                )
                # Using Nelder-Mead as it's derivative-free. For smoother problems, 'BFGS' or 'L-BFGS-B' could be faster.
                res = minimize(obj_func_for_scipy, initial_guess_x_p, method='Nelder-Mead', tol=1e-6)
                if res.success:
                    self.x_vars[p_idx, :] = res.x
                else:
                    # print(f"Warning: x_p-update for p={p_idx} at iter {k_iter} did not converge. Message: {res.message}")
                    pass # Keep old value or handle error

            # 2. {y_p}-update (diversity promoting step)
            # This solves A_y * y_l = rhs_l for each dimension l of y,
            # where A_y is self.A_y_update_matrix.
            if self.P > 0: 
                # c_val_p = x_p^{k+1} + u_p^k for each p
                c_val_for_y_update = self.x_vars + self.u_vars 

                for l_dim in range(self.dim_x): # Solve for each dimension of y independently
                    rhs_l_dim = self.rho * c_val_for_y_update[:, l_dim]
                    if self.A_inv_y_update is not None:
                        self.y_vars[:, l_dim] = np.dot(self.A_inv_y_update, rhs_l_dim)
                    elif self.A_y_update_matrix.size > 0 and self.P > 0 : # Fallback to least squares if P > 0
                        solution_l_dim, _, _, _ = np.linalg.lstsq(self.A_y_update_matrix, rhs_l_dim, rcond=None)
                        self.y_vars[:, l_dim] = solution_l_dim
            
            # 3. u_p-updates (parallelizable)
            if self.P > 0:
                self.u_vars += (self.x_vars - self.y_vars)

            # Calculate residuals for stopping criteria
            if self.P > 0:
                # Normalized primal residual (per variable component)
                primal_res = np.linalg.norm(self.x_vars - self.y_vars) / (np.sqrt(self.P * self.dim_x) + 1e-9)
                # Normalized dual residual (per variable component)
                dual_res = np.linalg.norm(self.rho * (self.y_vars - y_vars_old_for_dual_res)) / (np.sqrt(self.P * self.dim_x) + 1e-9)
            else:
                primal_res = 0.0
                dual_res = 0.0
            
            history['primal_res'].append(primal_res)
            history['dual_res'].append(dual_res)

            if k_iter % 20 == 0:
                print(f"Iter {k_iter}: Primal Res = {primal_res:.4e}, Dual Res = {dual_res:.4e}")

            if self.P > 0 and primal_res < self.tol and dual_res < self.tol:
                print(f"Converged at iteration {k_iter}.")
                break
        
        if k_iter == self.max_iters - 1 and not (self.P > 0 and primal_res < self.tol and dual_res < self.tol):
            print(f"Reached max iterations ({self.max_iters}). Primal Res = {primal_res:.4e}, Dual Res = {dual_res:.4e}")

        final_solutions = self.y_vars # y_vars are chosen as they incorporate the diversity term
        final_obj_values = np.array([self.F_func(final_solutions[p,:]) for p in range(self.P)]) if self.P > 0 else np.array()
        
        return final_solutions, final_obj_values, history

# --- Example Usage ---
if __name__ == "__main__":
    P_candidates = 100       # Desired number of Pareto optimal solutions
    dim_decision_vars = 2   # Dimension of x (e.g., x = [x0, x1])
    num_objectives = 2      # Number of objectives in F(x)

    rho_admm = 10.0         # ADMM penalty parameter
    beta_diversity = 0.5    # Diversity weight (for y-update)
    alpha_asf_param = 0.01  # ASF augmentation factor (small positive)
    
    max_admm_iters = 200
    convergence_tol = 1e-3

    # Aspiration levels will be scaled reference directions.
    # Scale factor for aspiration levels (e.g., based on expected objective range)
    aspiration_scale = 4.0 

    # Range for initializing x variables: (min_val, max_val)
    initial_x_val_range = (0.0, 2.0)


    print("Initializing SO-MO-ADMM...")
    somo_admm_solver = SOMO_ADMM(
        F_func=example_objective_functions,
        P=P_candidates,
        dim_x=dim_decision_vars,
        num_obj=num_objectives,
        rho=rho_admm,
        beta=beta_diversity,
        alpha_asf=alpha_asf_param,
        max_iters=max_admm_iters,
        tol=convergence_tol,
        aspiration_scaling_factor=aspiration_scale,
        initial_x_range=initial_x_val_range
    )

    print(f"Running SO-MO-ADMM for P={somo_admm_solver.P} candidates...") # P might have been adjusted
    final_sols_x_decision_space, final_sols_obj_space, run_history = somo_admm_solver.solve()

    print("\n--- Results ---")
    if final_sols_x_decision_space.size > 0 :
        print(f"Found {final_sols_x_decision_space.shape} potential Pareto optimal solutions.")
    else:
        print("Found 0 potential Pareto optimal solutions.")

    # Plotting the results (Objective Space)
    if final_sols_obj_space.size > 0 and num_objectives == 2:
        plt.figure(figsize=(10, 8))
        plt.scatter(final_sols_obj_space[:, 0], final_sols_obj_space[:, 1], c='blue', label='Found Pareto Solutions (SO-MO-ADMM)')
        
        # True Pareto front for the example problem:
        # Decision space: x0 = t, x1 = t for t in 
        # Objective space: f1 = 2*t^2, f2 = 2*(t-2)^2
        t_vals = np.linspace(0, 2, 100)
        true_f1 = 2 * t_vals**2
        true_f2 = 2 * (t_vals - 2)**2
        plt.plot(true_f1, true_f2, 'r--', label='True Pareto Front (Example Problem)')

        plt.xlabel('Objective 1: $f_1(x)$')
        plt.ylabel('Objective 2: $f_2(x)$')
        plt.title('Pareto Front Approximation using SO-MO-ADMM')
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig('expt4.png')
        # Plot residuals if history is not empty
        if run_history['primal_res']:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(run_history['primal_res'], label='Primal Residual')
            plt.xlabel('Iteration')
            plt.ylabel('Normalized Residual Norm')
            plt.title('Primal Residual Convergence')
            plt.yscale('log')
            plt.grid(True)
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(run_history['dual_res'], label='Dual Residual')
            plt.xlabel('Iteration')
            plt.ylabel('Normalized Residual Norm')
            plt.title('Dual Residual Convergence')
            plt.yscale('log')
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            plt.show()

    elif final_sols_obj_space.size > 0:
        print("Plotting is only configured for 2 objectives in this example.")
    elif somo_admm_solver.P == 0:
        print("No solutions to plot as P=0.")