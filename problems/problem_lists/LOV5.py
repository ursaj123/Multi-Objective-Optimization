import numpy as np

class LOV5:
    def __init__(self, n=3, lb=-2., ub=2., g_type=('zero', {}), num_pf_samples_dim=15):
        r"""
        LOV5 Problem (Lovison's Test Problem No. 5) - Strict Structure

        Objective Functions:
        f1(x) = (sqrt(2)/2)*x0 + (sqrt(2)/2)*f_aux(x)
        f2(x) = -(sqrt(2)/2)*x0 + (sqrt(2)/2)*f_aux(x)

        where x = (x0, x1, x2)^T
        f_aux(x) = g((x0,x1,x2)^T, M, p0, sigma0) + g((x0,x1,0.5*x2)^T, M, p1, sigma1)
        g(u, M_mat, p_vec, sigma_val) = sqrt(2*pi/sigma_val) * exp( -(u-p_vec)^T @ M_mat @ (u-p_vec) / sigma_val^2 )
        """
        self.m = 2  # Number of objectives
        if n != 3:
            raise ValueError("LOV5 problem is defined for n=3 variables.")
        self.n = n
        self.lb_val = lb # Store single value if uniform
        self.ub_val = ub # Store single value if uniform
        self.lb = np.array([lb] * self.n)
        self.ub = np.array([ub] * self.n)
        self.bounds = tuple([(self.lb[i], self.ub[i]) for i in range(self.n)])
        self.constraints = []
        self.g_type = g_type

        # Problem-specific constants
        self.p0 = np.array([0.0, 0.15, 0.0])
        self.p1 = np.array([0.0, -1.1, 0.0])
        self.M_mat = np.array([
            [-1.0, -0.03, 0.011],
            [-0.03, -1.0,  0.07 ],
            [0.011, 0.07, -1.01]
        ])
        self.sigma0 = 0.35
        self.sigma1 = 3.0
        self.sqrt2_div2 = np.sqrt(2) / 2.0

        self.num_pf_samples_dim = num_pf_samples_dim # Used in calculate_optimal_pareto_front
        self.true_pareto_front = self.calculate_optimal_pareto_front() # Renamed from _calculate...
        if hasattr(self, 'true_pareto_front') and self.true_pareto_front.size > 0:
            self.ref_point = np.max(self.true_pareto_front, axis=0) + 1e-4
        else:
            print("Warning: LOV5 Pareto front sampling might be empty or sparse. Using a generic ref point.")
            self.ref_point = np.array([25.0, 25.0]) # Adjusted fallback

    def f1(self, x_vars):
        x_vars_np = np.asarray(x_vars)
        
        # Component g0_val from f_aux
        u0 = x_vars_np
        diff_u0_p0 = u0 - self.p0
        exponent_g0 = -np.dot(diff_u0_p0, self.M_mat @ diff_u0_p0) / (self.sigma0**2)
        exponent_g0 = min(exponent_g0, 700) # Cap exponent
        g0_val = np.sqrt(2 * np.pi / self.sigma0) * np.exp(exponent_g0)

        # Component g1_val from f_aux
        u1 = np.array([x_vars_np[0], x_vars_np[1], 0.5 * x_vars_np[2]])
        diff_u1_p1 = u1 - self.p1
        exponent_g1 = -np.dot(diff_u1_p1, self.M_mat @ diff_u1_p1) / (self.sigma1**2)
        exponent_g1 = min(exponent_g1, 700) # Cap exponent
        g1_val = np.sqrt(2 * np.pi / self.sigma1) * np.exp(exponent_g1)
        
        f_aux_val = g0_val + g1_val
        return self.sqrt2_div2 * x_vars_np[0] + self.sqrt2_div2 * f_aux_val

    def grad_f1(self, x_vars):
        x_vars_np = np.asarray(x_vars)
        K0 = -2.0 / (self.sigma0**2)
        K1 = -2.0 / (self.sigma1**2)

        # --- Calculations for g0 and its gradient ---
        u0 = x_vars_np
        diff_u0_p0 = u0 - self.p0
        exponent_g0 = -np.dot(diff_u0_p0, self.M_mat @ diff_u0_p0) / (self.sigma0**2)
        exponent_g0 = min(exponent_g0, 700)
        g0_val = np.sqrt(2 * np.pi / self.sigma0) * np.exp(exponent_g0)
        
        if g0_val < 1e-100: grad_g0_wrt_x = np.zeros_like(u0)
        else: grad_g0_wrt_x = g0_val * K0 * (self.M_mat @ diff_u0_p0)

        # --- Calculations for g1 and its gradient ---
        u1 = np.array([x_vars_np[0], x_vars_np[1], 0.5 * x_vars_np[2]])
        diff_u1_p1 = u1 - self.p1
        exponent_g1 = -np.dot(diff_u1_p1, self.M_mat @ diff_u1_p1) / (self.sigma1**2)
        exponent_g1 = min(exponent_g1, 700)
        g1_val = np.sqrt(2 * np.pi / self.sigma1) * np.exp(exponent_g1)

        if g1_val < 1e-100: grad_g1_at_u1 = np.zeros_like(u1)
        else: grad_g1_at_u1 = g1_val * K1 * (self.M_mat @ diff_u1_p1)
        
        # Jacobian J_u1 = diag(1, 1, 0.5)
        # grad_g1_wrt_x = J_u1^T @ grad_g1_at_u1 (J_u1 is diagonal)
        grad_g1_wrt_x = np.array([grad_g1_at_u1[0], grad_g1_at_u1[1], 0.5 * grad_g1_at_u1[2]])
        
        grad_f_aux_val = grad_g0_wrt_x + grad_g1_wrt_x
        
        grad_total = self.sqrt2_div2 * grad_f_aux_val
        grad_total[0] += self.sqrt2_div2
        return grad_total

    def hess_f1(self, x_vars):
        x_vars_np = np.asarray(x_vars)
        K0 = -2.0 / (self.sigma0**2)
        K1 = -2.0 / (self.sigma1**2)

        # --- Calculations for g0, its gradient, and its Hessian ---
        u0 = x_vars_np
        diff_u0_p0 = u0 - self.p0
        exponent_g0 = -np.dot(diff_u0_p0, self.M_mat @ diff_u0_p0) / (self.sigma0**2)
        exponent_g0 = min(exponent_g0, 700)
        g0_val = np.sqrt(2 * np.pi / self.sigma0) * np.exp(exponent_g0)

        if g0_val < 1e-100:
            grad_g0_wrt_x = np.zeros_like(u0)
            hess_g0_wrt_x = np.zeros((self.n, self.n))
        else:
            grad_g0_wrt_x = g0_val * K0 * (self.M_mat @ diff_u0_p0)
            hess_g0_wrt_x = (np.outer(grad_g0_wrt_x, grad_g0_wrt_x) / g0_val) + (g0_val * K0 * self.M_mat)
            
        # --- Calculations for g1, its gradient, and its Hessian ---
        u1 = np.array([x_vars_np[0], x_vars_np[1], 0.5 * x_vars_np[2]])
        diff_u1_p1 = u1 - self.p1
        exponent_g1 = -np.dot(diff_u1_p1, self.M_mat @ diff_u1_p1) / (self.sigma1**2)
        exponent_g1 = min(exponent_g1, 700)
        g1_val = np.sqrt(2 * np.pi / self.sigma1) * np.exp(exponent_g1)

        if g1_val < 1e-100:
            grad_g1_at_u1 = np.zeros_like(u1)
            hess_g1_at_u1 = np.zeros((self.n, self.n)) # u1 has self.n components
        else:
            grad_g1_at_u1 = g1_val * K1 * (self.M_mat @ diff_u1_p1) # This is 3x1
            hess_g1_at_u1 = (np.outer(grad_g1_at_u1, grad_g1_at_u1) / g1_val) + (g1_val * K1 * self.M_mat)
        
        J_u1 = np.diag([1.0, 1.0, 0.5])
        # hess_g1_wrt_x = J_u1^T @ hess_g1_at_u1 @ J_u1
        hess_g1_wrt_x = J_u1 @ hess_g1_at_u1 @ J_u1 # Since J_u1 is symmetric

        hess_f_aux_val = hess_g0_wrt_x + hess_g1_wrt_x
        return self.sqrt2_div2 * hess_f_aux_val

    def f2(self, x_vars):
        x_vars_np = np.asarray(x_vars)
        
        # Component g0_val from f_aux (Identical to f1)
        u0 = x_vars_np
        diff_u0_p0 = u0 - self.p0
        exponent_g0 = -np.dot(diff_u0_p0, self.M_mat @ diff_u0_p0) / (self.sigma0**2)
        exponent_g0 = min(exponent_g0, 700)
        g0_val = np.sqrt(2 * np.pi / self.sigma0) * np.exp(exponent_g0)

        # Component g1_val from f_aux (Identical to f1)
        u1 = np.array([x_vars_np[0], x_vars_np[1], 0.5 * x_vars_np[2]])
        diff_u1_p1 = u1 - self.p1
        exponent_g1 = -np.dot(diff_u1_p1, self.M_mat @ diff_u1_p1) / (self.sigma1**2)
        exponent_g1 = min(exponent_g1, 700)
        g1_val = np.sqrt(2 * np.pi / self.sigma1) * np.exp(exponent_g1)
        
        f_aux_val = g0_val + g1_val
        return -self.sqrt2_div2 * x_vars_np[0] + self.sqrt2_div2 * f_aux_val

    def grad_f2(self, x_vars):
        x_vars_np = np.asarray(x_vars)
        K0 = -2.0 / (self.sigma0**2)
        K1 = -2.0 / (self.sigma1**2)

        # --- Calculations for g0 and its gradient (Identical to grad_f1) ---
        u0 = x_vars_np
        diff_u0_p0 = u0 - self.p0
        exponent_g0 = -np.dot(diff_u0_p0, self.M_mat @ diff_u0_p0) / (self.sigma0**2)
        exponent_g0 = min(exponent_g0, 700)
        g0_val = np.sqrt(2 * np.pi / self.sigma0) * np.exp(exponent_g0)
        
        if g0_val < 1e-100: grad_g0_wrt_x = np.zeros_like(u0)
        else: grad_g0_wrt_x = g0_val * K0 * (self.M_mat @ diff_u0_p0)

        # --- Calculations for g1 and its gradient (Identical to grad_f1) ---
        u1 = np.array([x_vars_np[0], x_vars_np[1], 0.5 * x_vars_np[2]])
        diff_u1_p1 = u1 - self.p1
        exponent_g1 = -np.dot(diff_u1_p1, self.M_mat @ diff_u1_p1) / (self.sigma1**2)
        exponent_g1 = min(exponent_g1, 700)
        g1_val = np.sqrt(2 * np.pi / self.sigma1) * np.exp(exponent_g1)

        if g1_val < 1e-100: grad_g1_at_u1 = np.zeros_like(u1)
        else: grad_g1_at_u1 = g1_val * K1 * (self.M_mat @ diff_u1_p1)
        
        grad_g1_wrt_x = np.array([grad_g1_at_u1[0], grad_g1_at_u1[1], 0.5 * grad_g1_at_u1[2]])
        
        grad_f_aux_val = grad_g0_wrt_x + grad_g1_wrt_x
        
        grad_total = self.sqrt2_div2 * grad_f_aux_val
        grad_total[0] -= self.sqrt2_div2 # Difference from grad_f1
        return grad_total

    def hess_f2(self, x_vars):
        # The Hessian of f_aux part is identical to hess_f1
        # The term -sqrt(2)/2 * x0 has zero Hessian
        return self.hess_f1(x_vars) # Re-uses the full calculation in hess_f1, which is correct
                                    # as only the constant term differs in gradient, not Hessian structure.
                                    # No, this is not quite right. hess_f1 itself returns sqrt2_div2 * hess_f_aux.
                                    # So this should be fine.
                                    # Let's be explicit:
        x_vars_np = np.asarray(x_vars)
        K0 = -2.0 / (self.sigma0**2)
        K1 = -2.0 / (self.sigma1**2)

        # --- Calculations for g0, its gradient, and its Hessian ---
        u0 = x_vars_np
        diff_u0_p0 = u0 - self.p0
        exponent_g0 = -np.dot(diff_u0_p0, self.M_mat @ diff_u0_p0) / (self.sigma0**2)
        exponent_g0 = min(exponent_g0, 700)
        g0_val = np.sqrt(2 * np.pi / self.sigma0) * np.exp(exponent_g0)

        if g0_val < 1e-100:
            grad_g0_wrt_x = np.zeros_like(u0) # Not strictly needed for Hessian if g0 is zero
            hess_g0_wrt_x = np.zeros((self.n, self.n))
        else:
            grad_g0_wrt_x = g0_val * K0 * (self.M_mat @ diff_u0_p0)
            hess_g0_wrt_x = (np.outer(grad_g0_wrt_x, grad_g0_wrt_x) / g0_val) + (g0_val * K0 * self.M_mat)
            
        # --- Calculations for g1, its gradient, and its Hessian ---
        u1 = np.array([x_vars_np[0], x_vars_np[1], 0.5 * x_vars_np[2]])
        diff_u1_p1 = u1 - self.p1
        exponent_g1 = -np.dot(diff_u1_p1, self.M_mat @ diff_u1_p1) / (self.sigma1**2)
        exponent_g1 = min(exponent_g1, 700)
        g1_val = np.sqrt(2 * np.pi / self.sigma1) * np.exp(exponent_g1)

        if g1_val < 1e-100:
            grad_g1_at_u1 = np.zeros_like(u1) # Not strictly needed
            hess_g1_at_u1 = np.zeros((self.n, self.n))
        else:
            grad_g1_at_u1 = g1_val * K1 * (self.M_mat @ diff_u1_p1)
            hess_g1_at_u1 = (np.outer(grad_g1_at_u1, grad_g1_at_u1) / g1_val) + (g1_val * K1 * self.M_mat)
        
        J_u1 = np.diag([1.0, 1.0, 0.5])
        hess_g1_wrt_x = J_u1 @ hess_g1_at_u1 @ J_u1

        hess_f_aux_val = hess_g0_wrt_x + hess_g1_wrt_x
        return self.sqrt2_div2 * hess_f_aux_val


    def calculate_optimal_pareto_front(self):
        """ Samples points in decision space, evaluates objectives, and filters for non-dominated set."""
        
        # Local helper function for non-dominated check (not a class method)
        def is_dominated(p1_obj_local, p2_obj_local):
            """Checks if p2_obj_local dominates p1_obj_local."""
            return np.all(p2_obj_local <= p1_obj_local) and np.any(p2_obj_local < p1_obj_local)

        # Local helper function for filtering (not a class method)
        def get_non_dominated_set_local(objectives_list_local):
            if not objectives_list_local:
                return np.array([])
            
            obj_array_local = np.array(objectives_list_local)
            num_points_local = obj_array_local.shape[0]
            is_dominated_flags_local = np.zeros(num_points_local, dtype=bool)

            for i in range(num_points_local):
                if is_dominated_flags_local[i]:
                    continue
                for j in range(num_points_local):
                    if i == j:
                        continue
                    # if is_dominated_flags_local[j]: # Can't use this optimization effectively here
                    #    continue
                    if is_dominated(obj_array_local[i], obj_array_local[j]): # if j dominates i
                        is_dominated_flags_local[i] = True
                        break 
            return obj_array_local[~is_dominated_flags_local]

        if self.num_pf_samples_dim <= 0:
            return np.array([]) # Return empty if no sampling
            
        num_samples_per_dim = self.num_pf_samples_dim
        x0_coords = np.linspace(self.lb[0], self.ub[0], num_samples_per_dim)
        x1_coords = np.linspace(self.lb[1], self.ub[1], num_samples_per_dim)
        x2_coords = np.linspace(self.lb[2], self.ub[2], num_samples_per_dim)

        all_objectives = []
        # Limit total points to avoid excessive computation if num_samples_per_dim is too large
        # For instance, cap at 50,000 points. (37^3 is ~50k)
        if num_samples_per_dim**self.n > 50000 and self.n == 3: # Rough check
            print(f"Warning: num_pf_samples_dim={num_samples_per_dim} is large, may be slow. Consider reducing.")
            # Potentially reduce num_samples_per_dim here if it's extremely large.

        for x0_val in x0_coords:
            for x1_val in x1_coords:
                for x2_val in x2_coords:
                    x_decision = np.array([x0_val, x1_val, x2_val])
                    # Need to call evaluate_f to get both objectives
                    obj_vals = np.array([self.f1(x_decision), self.f2(x_decision)]) 
                    all_objectives.append(obj_vals)
        
        if not all_objectives:
            return np.array([])
            
        non_dominated_objectives = get_non_dominated_set_local(all_objectives)
        return non_dominated_objectives

    def evaluate_f(self, x):
        return np.array([self.f1(x), self.f2(x)])

    def evaluate_gradients_f(self, x):
        return np.array([self.grad_f1(x), self.grad_f2(x)])

    def evaluate_hessians_f(self, x):
        return np.array([self.hess_f1(x), self.hess_f2(x)])

    def evaluate_g(self, z): # z is a dummy here if g_type is 'zero'
        if self.g_type[0] == 'zero':
            return np.zeros(self.m)
        else:
            raise NotImplementedError(f"g_type '{self.g_type[0]}' not implemented.")

    def evaluate(self, x, z):
        f_values = self.evaluate_f(x)
        g_values = self.evaluate_g(z)
        return f_values + g_values

    def f_list(self):
        return [
            (self.f1, self.grad_f1, self.hess_f1),
            (self.f2, self.grad_f2, self.hess_f2)
        ]

