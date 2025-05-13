import numpy as np

class LOV6:
    def __init__(self, n=6, g_type=('zero', {}), num_pf_samples=100):
        r"""
        LOV6 Problem (Lovison's Test Problem No. 6) - Strict Structure

        Objective Functions:
        f1(x) = x1
        f2(x) = 1 - sqrt(x1) - x1*sin(10*pi*x1) + sum_{i=2 to 6} (x_i^2)
        """
        self.m = 2  # Number of objectives
        if n != 6:
            raise ValueError("LOV6 problem is defined for n=6 variables.")
        self.n = n
        
        self.lb = np.array([0.1] + [-0.16] * (self.n - 1))
        self.ub = np.array([0.425] + [0.16] * (self.n - 1))
        
        self.bounds = tuple([(self.lb[i], self.ub[i]) for i in range(self.n)])
        self.constraints = []
        self.g_type = g_type # g_type for G(z) part
        
        self.num_pf_samples = num_pf_samples # Used in calculate_optimal_pareto_front
        self.true_pareto_front = self.calculate_optimal_pareto_front() # Renamed
        if hasattr(self, 'true_pareto_front') and self.true_pareto_front.size > 0:
            self.ref_point = np.max(self.true_pareto_front, axis=0) + 1e-4
        else:
             print("Warning: LOV6 Pareto front sampling might be empty. Using a generic ref point.")
             # Estimate based on f1 bounds and typical f2 range for x_i=0
             f2_at_lb_x1 = 1.0 - np.sqrt(self.lb[0]) - self.lb[0] * np.sin(10 * np.pi * self.lb[0])
             f2_at_ub_x1 = 1.0 - np.sqrt(self.ub[0]) - self.ub[0] * np.sin(10 * np.pi * self.ub[0])
             # Max possible f2 is when sin is -1, around 1-sqrt(0.1)+0.1 = 1-0.316+0.1 ~ 0.784
             # Min possible f2 is when sin is +1, around 1-sqrt(0.425)-0.425 = 1-0.65-0.425 ~ -0.075
             self.ref_point = np.array([self.ub[0] + 1e-4, max(f2_at_lb_x1, f2_at_ub_x1, 0.8) + 1e-4])


    def f1(self, x_vars):
        return x_vars[0]

    def grad_f1(self, x_vars):
        grad = np.zeros(self.n)
        grad[0] = 1.0
        return grad

    def hess_f1(self, x_vars):
        return np.zeros((self.n, self.n))

    def f2(self, x_vars):
        x_vars_np = np.asarray(x_vars)
        x1 = x_vars_np[0]
        sum_sq_others = np.sum(x_vars_np[1:]**2) # Corrected variable name
        val = 1.0 - np.sqrt(x1) - x1 * np.sin(10 * np.pi * x1) + sum_sq_others
        return val

    def grad_f2(self, x_vars):
        x_vars_np = np.asarray(x_vars)
        grad = np.zeros(self.n)
        x1 = x_vars_np[0]
        
        # Bounds for x1 are [0.1, 0.425], so x1 is positive.
        grad_x1_part = -0.5 / np.sqrt(x1) - (np.sin(10 * np.pi * x1) + x1 * np.cos(10 * np.pi * x1) * 10 * np.pi)
        grad[0] = grad_x1_part
        
        for i in range(1, self.n): 
            grad[i] = 2 * x_vars_np[i]
            
        return grad

    def hess_f2(self, x_vars):
        x_vars_np = np.asarray(x_vars)
        hess = np.zeros((self.n, self.n))
        x1 = x_vars_np[0]
        
        # Bounds for x1 are [0.1, 0.425], so x1 is positive and x1**(-1.5) is safe.
        hess_x1_x1 = 0.25 * (x1**(-1.5)) \
                     - 20 * np.pi * np.cos(10 * np.pi * x1) \
                     + (10 * np.pi)**2 * x1 * np.sin(10 * np.pi * x1)
        hess[0,0] = hess_x1_x1
        
        for i in range(1, self.n): 
            hess[i,i] = 2.0
            
        return hess

    def calculate_optimal_pareto_front(self): # Renamed from _calculate...
        if self.num_pf_samples <= 0:
            return np.array([])
            
        x1_values_pf = np.linspace(self.lb[0], self.ub[0], self.num_pf_samples)
        pf_points = []
        for x1_val_pf in x1_values_pf:
            f1_val = x1_val_pf
            # For Pareto front, x2 to x6 are 0
            f2_val = 1.0 - np.sqrt(x1_val_pf) - x1_val_pf * np.sin(10 * np.pi * x1_val_pf) 
            pf_points.append([f1_val, f2_val])
        return np.array(pf_points)

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