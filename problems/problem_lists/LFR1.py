import numpy as np

class LFR1:
    def __init__(self, n=10, m=2, lb=-5., ub=5., g_type=('zero', {}), num_pf_samples=100):
        r"""
        LFR1 Problem

        Objective Functions (using 0-based indexing for x_vars: x_0, ..., x_{n-1}):
        Let S(x) = sum_{k=0 to n-1} (k+1)*x_k
        f_i(x) = (i_val * S(x) - 1)^2, where i_val is (objective_index + 1)

        Default n_features = 10, default n_objectives = 4.
        Default Bounds: x_k in [-5, 5]
        """
        self.n = n  # Number of decision variables
        self.m = m  # Number of objectives
        
        self.lb_val = lb  # Scalar lower bound for each x_k
        self.ub_val = ub  # Scalar upper bound for each x_k

        self.lb_arr = np.array([lb] * self.n) 
        self.ub_arr = np.array([ub] * self.n)
        
        self.bounds = tuple([(self.lb_arr[k], self.ub_arr[k]) for k in range(self.n)])
        self.constraints = [] 
        self.g_type = g_type
        
        # Precompute j_vector (coefficients for S(x)) and base Hessian matrix
        # j_vector corresponds to (1, 2, ..., n) for x_0, x_1, ..., x_{n-1}
        self.j_vector = np.arange(1, self.n + 1)
        self.hess_base_matrix = np.outer(self.j_vector, self.j_vector)

        # Dynamically create f_i, grad_f_i, hess_f_i methods and store them
        self._f_actual_list = []
        for i_idx in range(self.m):
            i_one_based = i_idx + 1 # The 'i' in the problem formula

            # Define functions with captured i_one_based
            # Note: Default arguments capture values at definition time
            def current_f(x_vars, _i_val=i_one_based, _j_vec=self.j_vector):
                Sx = np.dot(_j_vec, np.asarray(x_vars))
                return (_i_val * Sx - 1)**2

            def current_grad_f(x_vars, _i_val=i_one_based, _j_vec=self.j_vector):
                Sx = np.dot(_j_vec, np.asarray(x_vars))
                factor = 2 * _i_val * (_i_val * Sx - 1)
                return factor * _j_vec
            
            def current_hess_f(x_vars, _i_val=i_one_based, _hess_base=self.hess_base_matrix):
                # x_vars is not used as Hessian is constant for a given i_val
                factor = 2 * _i_val**2
                return factor * _hess_base

            self._f_actual_list.append((current_f, current_grad_f, current_hess_f))
            
            # Optionally, make them accessible as self.f1, self.grad_f1 etc.
            # setattr(self, f"f{i_one_based}", current_f)
            # setattr(self, f"grad_f{i_one_based}", current_grad_f)
            # setattr(self, f"hess_f{i_one_based}", current_hess_f)

        self.num_pf_samples = num_pf_samples
        self.true_pareto_front = self.calculate_optimal_pareto_front()

        self.l1_ratios, self.l1_shifts = [], []
        if self.g_type[0]=='L1':
            for i in range(self.m):
                self.l1_ratios.append(1/((i+1)*self.m))
                self.l1_shifts.append(i)
            self.l1_ratios = np.array(self.l1_ratios)
            self.l1_shifts = np.array(self.l1_shifts)
        
        if hasattr(self, 'true_pareto_front') and self.true_pareto_front.size > 0:
            self.ref_point = np.max(self.true_pareto_front, axis=0) + 1e-4
        else:
             print(f"Warning: LFR1 Pareto front sampling (m={self.m}, n={self.n}) might be empty or result in NaNs. Using a generic ref point.")
             # Estimate based on Y=0, where f_i = 1
             self.ref_point = np.ones(self.m) * 1.1 # Slightly above (1,1,...)

    # Individual f_i, grad_f_i, hess_f_i are not explicitly defined as f1, f2...
    # They are accessed via f_list or evaluate_f methods.
    # If explicit self.f1 etc. are needed, uncomment setattr lines in __init__

    def calculate_optimal_pareto_front(self):
        """
        Calculates the true Pareto front for the LFR1 problem.
        The Pareto front is a curve parameterized by Y = S(x).
        f_i(Y) = (i_val * Y - 1)^2
        """
        if self.num_pf_samples <= 0:
            return np.array([])

        sum_coeffs = np.sum(self.j_vector) # sum_{k=0 to n-1} (k+1) = n(n+1)/2
        
        # Determine range of Y = S(x)
        # Since j_vector components are all positive:
        Y_min = self.lb_val * sum_coeffs
        Y_max = self.ub_val * sum_coeffs

        if Y_min > Y_max: # Should not happen if lb_val <= ub_val
            Y_min, Y_max = Y_max, Y_min # Swap if lb_val > ub_val (though unusual)

        if self.num_pf_samples == 1:
            # Handle edge case: sample at midpoint or one of the ends
            Y_values = np.array([(Y_min + Y_max) / 2.0])
        else:
            Y_values = np.linspace(Y_min, Y_max, self.num_pf_samples)
        
        pareto_front_points = []
        for Y_val in Y_values:
            obj_vector = np.zeros(self.m)
            for i_idx in range(self.m):
                i_one_based = i_idx + 1
                val = (i_one_based * Y_val - 1)**2
                obj_vector[i_idx] = val
            pareto_front_points.append(obj_vector)
        
        pf_array = np.array(pareto_front_points)
        if np.isnan(pf_array).any() or np.isinf(pf_array).any():
            print(f"Warning: NaN or Inf encountered in LFR1 Pareto front calculation. Y range: [{Y_min}, {Y_max}]")
            # Return empty or a safe default if issues occur
            return np.array([])
            
        return pf_array

    def evaluate_f(self, x_vars):
        """
        Evaluates all m objective functions at point x_vars.
        """
        x = np.asarray(x_vars)
        obj_values = np.zeros(self.m)
        # Sx = np.dot(self.j_vector, x) # Calculate S(x) once
        for i_idx in range(self.m):
            # i_one_based = i_idx + 1
            # obj_values[i_idx] = (i_one_based * Sx - 1)**2
            obj_values[i_idx] = self._f_actual_list[i_idx][0](x) # Call the stored function
        return obj_values

    def evaluate_gradients_f(self, x_vars):
        """
        Evaluates the gradients of all m objective functions at point x_vars.
        Returns a numpy array of shape (m, n).
        """
        x = np.asarray(x_vars)
        grad_values = np.zeros((self.m, self.n))
        # Sx = np.dot(self.j_vector, x) # Calculate S(x) once
        for i_idx in range(self.m):
            # i_one_based = i_idx + 1
            # factor = 2 * i_one_based * (i_one_based * Sx - 1)
            # grad_values[i_idx, :] = factor * self.j_vector
            grad_values[i_idx, :] = self._f_actual_list[i_idx][1](x) # Call the stored function
        return grad_values

    def evaluate_hessians_f(self, x_vars):
        """
        Evaluates the Hessians of all m objective functions at point x_vars.
        Returns a numpy array of shape (m, n, n).
        """
        # x_vars is not used as Hessians are constant for a given i_val
        hess_values = np.zeros((self.m, self.n, self.n))
        for i_idx in range(self.m):
            # i_one_based = i_idx + 1
            # factor = 2 * i_one_based**2
            # hess_values[i_idx, :, :] = factor * self.hess_base_matrix
            hess_values[i_idx, :, :] = self._f_actual_list[i_idx][2](x_vars) # Pass x_vars for consistent API
        return hess_values

    def evaluate_g(self, z_vars):
        """
        Evaluates the G(z) part of the objective F(x) + G(z).
        """
        if self.g_type[0] == 'zero':
            return np.zeros(self.m)
        elif self.g_type[0] == 'L1':
            res = np.zeros(self.m)
            for i in range(self.m):
                res[i] = np.linalg.norm((z-self.l1_shifts[i])*self.l1_ratios[i], ord=1)
            return res
        elif self.g_type[0] == 'indicator':
            pass
        else:
            pass
    def evaluate(self, x_vars, z_vars):
        """
        Evaluates the combined objective F(x) + G(z).
        """
        f_vals = self.evaluate_f(x_vars)
        g_vals = self.evaluate_g(z_vars)
        return f_vals + g_vals

    def f_list(self):
        """
        Returns a list of tuples, where each tuple contains (fi, grad_fi, hess_fi)
        for each objective function fi.
        """
        return self._f_actual_list

