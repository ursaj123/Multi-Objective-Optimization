import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from itertools import combinations

class ADMMMultiObjectiveSolver:
    def __init__(self, objectives, gradients, bounds, z_star=None, 
                 rho=1.0, max_iter=200, tol=1e-6):
        """
        Properly initialized class with complete method definitions
        """
        self.objectives = objectives
        self.gradients = gradients
        self.bounds = bounds
        self.num_objs = len(objectives)
        self.rho = rho
        self.max_iter = max_iter
        self.tol = tol
        self.z_star = self._compute_ideal_point() if z_star is None else z_star

    def _compute_ideal_point(self):
        """Compute ideal point by minimizing each objective individually"""
        z_star = np.zeros(self.num_objs)
        for i in range(self.num_objs):
            res = minimize(
                fun=self.objectives[i],
                x0=np.array([(low+high)/2 for (low, high) in self.bounds]),
                method='L-BFGS-B',
                bounds=self.bounds
            )
            z_star[i] = res.fun
        return z_star

    def _create_x_update_functions(self, z, lambda_):
        """Properly create objective and gradient functions for x-update"""
        def objective(x):
            total = 0.0
            for i in range(self.num_objs):
                f_i = self.objectives[i](x)
                total += -lambda_[i] * f_i + (self.rho/2) * (f_i - z[i])**2
            return total
        
        def gradient(x):
            grad = np.zeros_like(x)
            for i in range(self.num_objs):
                f_i = self.objectives[i](x)
                grad_f = self.gradients[i](x)
                grad += (-lambda_[i] + self.rho*(f_i - z[i])) * grad_f
            return grad
        
        return objective, gradient

    def solve(self, weights, x_init=None):
        """Robust solving with proper function initialization"""
        if x_init is None:
            x_init = np.array([np.random.uniform(low, high) for (low, high) in self.bounds])
        
        x = x_init.copy()
        z = np.array([f(x) for f in self.objectives])
        lambda_ = np.zeros(self.num_objs)
        
        for _ in range(self.max_iter):
            # Create fresh functions each iteration to avoid closure issues
            obj_func, grad_func = self._create_x_update_functions(z, lambda_)
            
            # x-update with bounds
            res = minimize(
                fun=obj_func,
                x0=x,
                method='L-BFGS-B',
                jac=grad_func,
                bounds=self.bounds
            )
            x_new = res.x
            f_x_new = np.array([f(x_new) for f in self.objectives])
            
            # z-update with Tchebycheff projection
            z_unconstrained = f_x_new - lambda_/self.rho
            t = np.max(weights * (z_unconstrained - self.z_star))
            z_new = np.minimum(z_unconstrained, self.z_star + t/weights)
            
            # Dual update
            lambda_new = lambda_ + self.rho * (z_new - f_x_new)
            
            # Check convergence
            if np.linalg.norm(z_new - z) < self.tol:
                break
                
            x, z, lambda_ = x_new, z_new, lambda_new
            
        return {'x': x, 'objectives': f_x_new}

def generate_pareto_front(solver, num_points=50, trials=5):
    """Improved Pareto front generation with diversity maintenance"""
    pareto_points = []
    
    # Generate weights using simplex projection
    if solver.num_objs == 2:
        weights = np.array([[w, 1-w] for w in np.linspace(0.01, 0.99, num_points)])
    else:
        weights = np.random.dirichlet(np.ones(solver.num_objs), size=num_points)
    
    for w in weights:
        for _ in range(trials):
            result = solver.solve(w/np.sum(w))
            if result is not None:
                pareto_points.append(result['objectives'])
    
    # Fast non-dominated sort
    pareto_front = []
    for point in pareto_points:
        dominated = False
        for other in pareto_points:
            if np.all(other <= point) and np.any(other < point):
                dominated = True
                break
        if not dominated:
            pareto_front.append(point)
    
    return np.array(sorted(pareto_front, key=lambda x: x[0]))

# =============================================
# Test Case: Convex Problem with Bounds
# =============================================


def zdt2(n_vars=30):
    """Returns objectives and gradients for ZDT2 problem with n variables"""
    # Objective functions
    def f1(x):
        return x[0]
    
    def f2(x):
        g = 1 + 9/(n_vars-1) * np.sum(x[1:])
        return g * (1 - (x[0]/g)**2)
    
    # Gradients
    def grad_f1(x):
        grad = np.zeros(n_vars)
        grad[0] = 1.0
        return grad
    
    def grad_f2(x):
        grad = np.zeros(n_vars)
        g = 1 + 9/(n_vars-1) * np.sum(x[1:])
        dg_dxi = 9/(n_vars-1)  # Derivative of g w.r.t. any x[i] (i > 0)
        
        # df2/dx0
        grad[0] = -2*x[0]/g
        
        # df2/dx1...dxn (CORRECTED)
        for i in range(1, n_vars):
            grad[i] = (1 - (x[0]/g)**2) * dg_dxi + (2 * x[0]**2 / g**2) * dg_dxi  # Fixed denominator
        
        return grad
    
    return [f1, f2], [grad_f1, grad_f2]


if __name__ == "__main__":
    # Define objectives and gradients
    # objectives = [
    #     lambda x: 0.5*np.linalg.norm(x)**2,
    #     lambda x: 0.5*np.linalg.norm(x - 1)**2
    # ]
    # gradients = [
    #     lambda x: x,
    #     lambda x: x - 1
    # ]
    # bounds = [(-2, 2)]*4  # 4-dimensional problem
    n_vars = 4
    bounds = [(0, 1)] * n_vars
    
    # Get objectives and gradients
    objectives, gradients = zdt2(n_vars)
    # Initialize solver
    solver = ADMMMultiObjectiveSolver(
        objectives=objectives,
        gradients=gradients,
        bounds=bounds,
        rho=0.8,
        max_iter=1000
    )
    
    # Generate Pareto front
    pf = generate_pareto_front(solver, num_points=1000, trials=3)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(pf[:, 0], pf[:, 1], c='r', edgecolor='k', alpha=0.7)
    plt.xlabel("f1(x)"), plt.ylabel("f2(x)")
    plt.title("Properly Implemented Pareto Front with Bounds")
    plt.grid(True)
    plt.savefig('expt2.png')
    plt.show()