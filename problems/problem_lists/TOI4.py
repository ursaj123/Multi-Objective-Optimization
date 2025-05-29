import numpy as np

class TOI4:
    def __init__(self, n=4, lb=-5., ub=5., g_type=('zero', {}), num_pf_samples=100):
        r"""
        TOI4 Problem (User Updated Formulation)

        Objective Functions:
        f1(x) = x1^2 + x2^2 + 1
        f2(x) = 0.5 * ((x1 - x2)^2 + (x3 - x4)^2) + 1

        Variables: x = (x1, x2, x3, x4)^T
        Default Bounds: x_i in [-5, 5] for i=0,1,2,3 (using 0-based indexing for x)
        """
        self.m = 2  # Number of objectives
        if n != 4:
            raise ValueError("TOI4 problem is defined for n=4 variables.")
        self.n = n
        
        # Store bounds for Pareto front calculation (U in analysis)
        self.lb_val = lb 
        self.ub_val = ub

        self.lb_arr = np.array([lb] * self.n) 
        self.ub_arr = np.array([ub] * self.n)
        
        self.bounds = tuple([(self.lb_arr[i], self.ub_arr[i]) for i in range(self.n)])
        self.constraints = [] 
        self.g_type = g_type
        
        self.num_pf_samples = num_pf_samples
        self.true_pareto_front = self.calculate_optimal_pareto_front()
        
        if hasattr(self, 'true_pareto_front') and self.true_pareto_front.size > 0:
            # For f2=1, max f1 = 2*ub_val^2+1
            # Ref point is slightly above the max values on the Pareto front
            max_f1_pf = np.max(self.true_pareto_front[:, 0])
            max_f2_pf = np.max(self.true_pareto_front[:, 1]) # Should be 1
            self.ref_point = np.array([max_f1_pf + 1e-4, max_f2_pf + 1e-4])
        else:
             # Fallback reference point
             print("Warning: TOI4 Pareto front sampling might be empty. Using a generic ref point.")
             # Max f1 approx 2*ub_val^2+1, Max f2 is 1
             f1_est_max = 2 * (self.ub_val**2) + 1.0 if self.ub_val >= 0 else 1.0 # k_a^2 is non-negative
             f2_est_max = 1.0
             self.ref_point = np.array([f1_est_max + 1.0, f2_est_max + 1.0]) # e.g. [52, 2] for ub=5


        self.l1_ratios, self.l1_shifts = [], []
        if self.g_type[0]=='L1':
            for i in range(self.m):
                self.l1_ratios.append(1/((i+1)*self.m))
                self.l1_shifts.append(i)
            self.l1_ratios = np.array(self.l1_ratios)
            self.l1_shifts = np.array(self.l1_shifts)

    def f1(self, x_vars):
        """
        Objective function f1(x) = x_0^2 + x_1^2 + 1
        (using 0-based indexing for x_vars)
        """
        x = np.asarray(x_vars)
        return x[0]**2 + x[1]**2 + 1.0

    def grad_f1(self, x_vars):
        """
        Gradient of f1(x).
        grad_f1 = [2*x_0, 2*x_1, 0, 0]^T
        """
        x = np.asarray(x_vars)
        grad = np.zeros(self.n)
        grad[0] = 2 * x[0]
        grad[1] = 2 * x[1]
        return grad

    def hess_f1(self, x_vars):
        """
        Hessian of f1(x).
        H_f1 = diag(2, 2, 0, 0)
        """
        # x_vars is not used as the Hessian is constant for non-zero elements.
        hess = np.zeros((self.n, self.n))
        hess[0,0] = 2.0
        hess[1,1] = 2.0
        return hess

    def f2(self, x_vars):
        """
        Objective function f2(x) = 0.5 * ((x_0 - x_1)^2 + (x_2 - x_3)^2) + 1
        (using 0-based indexing for x_vars)
        """
        x = np.asarray(x_vars)
        term1_sq_diff = (x[0] - x[1])**2
        term2_sq_diff = (x[2] - x[3])**2
        return 0.5 * (term1_sq_diff + term2_sq_diff) + 1.0

    def grad_f2(self, x_vars):
        """
        Gradient of f2(x).
        grad_f2[0] = x_0 - x_1
        grad_f2[1] = x_1 - x_0
        grad_f2[2] = x_2 - x_3
        grad_f2[3] = x_3 - x_2
        """
        x = np.asarray(x_vars)
        grad = np.zeros(self.n)
        grad[0] = x[0] - x[1]
        grad[1] = x[1] - x[0] # -(x[0] - x[1])
        grad[2] = x[2] - x[3]
        grad[3] = x[3] - x[2] # -(x[2] - x[3])
        return grad

    def hess_f2(self, x_vars):
        """
        Hessian of f2(x).
        H_f2 = [[ 1, -1,  0,  0],
                [-1,  1,  0,  0],
                [ 0,  0,  1, -1],
                [ 0,  0, -1,  1]]
        """
        # x_vars is not used as the Hessian is constant.
        hess = np.zeros((self.n, self.n))
        
        hess[0,0] =  1.0
        hess[0,1] = -1.0
        hess[1,0] = -1.0
        hess[1,1] =  1.0
        
        hess[2,2] =  1.0
        hess[2,3] = -1.0
        hess[3,2] = -1.0
        hess[3,3] =  1.0
        return hess

    def calculate_optimal_pareto_front(self):
        """
        Calculates the true Pareto front for the updated TOI4 problem.
        The Pareto front is:
        f2 = 1
        f1 in [1, 2*U^2 + 1], where U is the magnitude of the bound (e.g., ub_val if symmetric 0-centered)
        """
        if self.num_pf_samples <= 0:
            return np.array([])

        # U is the effective upper bound for |k_a| where x1=x2=k_a
        # Assuming symmetric bounds [-U, U], U = self.ub_val
        # If bounds are asymmetric, U would be min(|lb_val|, |ub_val|) if k_a can be negative,
        # or depends on how k_a^2 is maximized. For k_a^2, max is max(lb_val^2, ub_val^2).
        # Assuming bounds are like [-5, 5], so U = 5.
        # If bounds are [0,5], U=5. If [-2,5], U_for_sq = 5.
        # Let's use self.ub_val as U, assuming it's positive and symmetric around 0 or positive interval.
        
        # Max k_a^2. If bounds are [-U,U], then max k_a^2 is U^2.
        # If bounds are [L,U], max k_a^2 is max(L^2, U^2) if 0 is not in [L,U] or L,U have different signs.
        # If 0 is in [L,U], then max k_a^2 is max(L^2,U^2).
        # For simplicity, assuming bounds are symmetric [-U,U] or [0,U], so U = self.ub_val
        # This means k_a can be self.ub_val.
        # Or more generally, k_a is such that it's within its own bounds.
        # The variable x_0 (which is k_a) is bounded by self.lb_arr[0] and self.ub_arr[0].
        # So k_a^2 is in [0, max(self.lb_arr[0]^2, self.ub_arr[0]^2)] if 0 is between lb and ub for x0.
        # Or more simply, if k_a is in [L0, U0], then k_a^2 is in [min_sq, max_sq]
        # min_sq is 0 if L0 <= 0 <= U0, else min(L0^2, U0^2).
        # max_sq is max(L0^2, U0^2).
        
        # For f1 = 2*k_a^2+1, k_a is x_0.
        # Min k_a^2 is 0 if 0 is in [self.lb_arr[0], self.ub_arr[0]]. Otherwise min(lb0^2, ub0^2).
        # Max k_a^2 is max(self.lb_arr[0]^2, self.ub_arr[0]^2).
        
        min_k_sq = 0.0
        if self.lb_arr[0] > 0 or self.ub_arr[0] < 0: # 0 is not in the interval for x0
            min_k_sq = min(self.lb_arr[0]**2, self.ub_arr[0]**2)
        max_k_sq = max(self.lb_arr[0]**2, self.ub_arr[0]**2)

        f1_min_pf = 2 * min_k_sq + 1.0
        f1_max_pf = 2 * max_k_sq + 1.0
        
        if self.num_pf_samples == 1:
            f1_values_pf = np.array([f1_min_pf]) # Or an average, or a specific point
        else:
            f1_values_pf = np.linspace(f1_min_pf, f1_max_pf, self.num_pf_samples)
        
        f2_values_pf = np.ones(self.num_pf_samples) # f2 is always 1 on the Pareto front
        
        pareto_front_points = np.stack((f1_values_pf, f2_values_pf), axis=-1)
        return pareto_front_points

    def evaluate_f(self, x):
        """
        Evaluates all objective functions at point x.
        Returns a numpy array: [f1(x), f2(x)]
        """
        return np.array([self.f1(x), self.f2(x)])

    def evaluate_gradients_f(self, x):
        """
        Evaluates the gradients of all objective functions at point x.
        Returns a numpy array: [grad_f1(x), grad_f2(x)]
        """
        return np.array([self.grad_f1(x), self.grad_f2(x)])

    def evaluate_hessians_f(self, x):
        """
        Evaluates the Hessians of all objective functions at point x.
        Returns a numpy array: [hess_f1(x), hess_f2(x)]
        """
        return np.array([self.hess_f1(x), self.hess_f2(x)])

    def evaluate_g(self, z):
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

    def evaluate(self, x, z):
        """
        Evaluates the combined objective F(x) + G(z).
        """
        f_values = self.evaluate_f(x)
        g_values = self.evaluate_g(z)
        return f_values + g_values

    def f_list(self):
        """
        Returns a list of tuples, where each tuple contains (fi, grad_fi, hess_fi)
        for each objective function fi.
        """
        return [
            (self.f1, self.grad_f1, self.hess_f1),
            (self.f2, self.grad_f2, self.hess_f2)
        ]

