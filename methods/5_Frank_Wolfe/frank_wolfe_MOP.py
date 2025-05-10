import numpy as np
from scipy.optimize import linprog

class JOS1:
    """Implements the JOS1 multiobjective optimization problem from literature.
    
    The problem has two quadratic objectives:
    - f₁(x) = 0.5‖x‖²
    - f₂(x) = 0.5‖x - 2‖²
    
    where x ∈ ℝⁿ and typically has box constraints
    
    Attributes:
        m (int): Number of objectives (fixed at 2)
        n (int): Number of decision variables
        bounds (list): Box constraints for decision variables
    """
    
    def __init__(self, n=2, bounds=None):
        """Initialize JOS1 problem with specified dimension and constraints.
        
        Args:
            n (int): Number of decision variables
            bounds (list): List of (min, max) tuples for each variable.
                           Defaults to [-100, 100] for all variables.
        """
        self.m = 2
        self.n = n
        self.bounds = bounds or [(-100.0, 100.0)]*n

    def f1(self, x):
        """Compute first objective value: f₁(x) = 0.5‖x‖²
        
        Args:
            x (np.ndarray): Decision variable vector
            
        Returns:
            float: Objective value
        """
        return 0.5 * np.sum(x**2)

    def grad_f1(self, x):
        """Gradient of first objective: ∇f₁(x) = x
        
        Args:
            x (np.ndarray): Decision variable vector
            
        Returns:
            np.ndarray: Gradient vector
        """
        return x

    def f2(self, x):
        """Compute second objective value: f₂(x) = 0.5‖x - 2‖²
        
        Args:
            x (np.ndarray): Decision variable vector
            
        Returns:
            float: Objective value
        """
        return 0.5 * np.sum((x - 2)**2)

    def grad_f2(self, x):
        """Gradient of second objective: ∇f₂(x) = x - 2
        
        Args:
            x (np.ndarray): Decision variable vector
            
        Returns:
            np.ndarray: Gradient vector
        """
        return x - 2

    def f_list(self):
        """Get list of objective functions and their gradients
        
        Returns:
            list: Contains tuples of (objective function, gradient function)
        """
        return [(self.f1, self.grad_f1), (self.f2, self.grad_f2)]



class FrankWolfeMOP:
    """Implements the Generalized Conditional Gradient Method (Frank-Wolfe)
    for multiobjective composite optimization problems.
    
    As described in:
    "A generalized conditional gradient method for multiobjective composite optimization problems"
    
    Attributes:
        problem (JOS1): Multiobjective problem instance
        zeta (float): Armijo condition parameter (0 < ζ < 1)
        omega1 (float): Step size reduction lower bound (0 < ω₁ < ω₂ < 1)
        omega2 (float): Step size reduction upper bound
        max_iter (int): Maximum number of iterations
        tol (float): Convergence tolerance for gap function θ
    """
    def __init__(self, problem, zeta=1e-4, omega1=0.05, omega2=0.95,
                 max_iter=1000, tol=1e-4, verbose=True):  ## DEBUG added verbose
        """Initialize solver with algorithm parameters
        
        Args:
            problem (JOS1): Multiobjective optimization problem
            zeta (float): Armijo sufficient decrease parameter
            omega1 (float): Lower bound for step size reduction
            omega2 (float): Upper bound for step size reduction
            max_iter (int): Maximum allowed iterations
            tol (float): Convergence tolerance
        """
        self.problem = problem
        self.zeta = zeta
        self.omega1 = omega1
        self.omega2 = omega2
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose  ## DEBUG
        
    def solve_subproblem(self, x):
        """Solve the linear subproblem to find descent direction (Algorithm 1 Step 1)
        
        Computes:
            p(x) ∈ argmin_u max_j [g_j(u) - g_j(x) + ⟨∇h_j(x), u - x⟩]
            θ(x) = max_j [g_j(p(x)) - g_j(x) + ⟨∇h_j(x), p(x) - x⟩]
        
        Formulated as linear program (LP):
            min τ
            s.t. ⟨∇h_j(x), u⟩ - τ ≤ ⟨∇h_j(x), x⟩ - g_j(x) ∀j
                 lb ≤ u ≤ ub
        
        Args:
            x (np.ndarray): Current decision variable vector
            
        Returns:
            tuple: (p(x), θ(x)) solution and gap value
        """
        if self.verbose:  ## DEBUG
            print(f"\n## DEBUG: Solving subproblem at x = {x}")
            
        # ... (keep existing setup code) ...

        if self.verbose:  ## DEBUG
            print(f"## DEBUG: LP coefficients:")
            print(f"Objective: {c}")
            print(f"Constraint matrix:\n{A_ub}")
            print(f"RHS bounds: {b_ub}")
            print(f"Variable bounds: {bounds}")

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if self.verbose:  ## DEBUG
            if result.success:
                print(f"## DEBUG: Subproblem solved successfully")
                print(f"Solution: u = {result.x[:-1]}, τ = {result.x[-1]:.4e}")
            else:
                print(f"## DEBUG: Subproblem failed with status: {result.status}")
                print(f"Message: {result.message}")
                
        if not result.success:
            raise RuntimeError(f"Subproblem failed: {result.message}")

        return result.x[:-1], result.x[-1]

    def armijo_line_search(self, x, d, theta):
        """Armijo-type line search for step size selection (Section 4.1)
        
        Finds largest α ∈ (0,1] satisfying:
        F(x + αd) ≼ F(x) + ζαθe
        
        where e = (1,...,1) ∈ ℝ^m and θ < 0
        
        Args:
            x (np.ndarray): Current decision variables
            d (np.ndarray): Descent direction
            theta (float): Gap function value (θ < 0)
            
        Returns:
            float: Valid step size α
        """
        if self.verbose:  ## DEBUG
            print(f"\n## DEBUG: Starting Armijo line search")
            print(f"Initial direction: {d}")
            print(f"Initial θ: {theta:.4e}")

        alpha = 1.0
        f_list = self.problem.f_list()
        f_x = np.array([f(x) for f, _ in f_list])
        theta_abs = abs(theta)
        
        for i in range(100):
            x_new = x + alpha * d
            f_new = np.array([f(x_new) for f, _ in f_list])
            condition_met = np.all(f_new <= f_x - self.zeta * alpha * theta_abs)
            
            if self.verbose:  ## DEBUG
                print(f"## DEBUG: Trial {i+1}: α = {alpha:.4e}")
                print(f"New objectives: {f_new}")
                print(f"Required decrease: {self.zeta * alpha * theta_abs:.4e}")
                print(f"Condition met: {condition_met}")

            if condition_met:
                if self.verbose:  ## DEBUG
                    print(f"## DEBUG: Accepted α = {alpha:.4e} after {i+1} trials")
                return alpha
                
            alpha = max(self.omega1 * alpha, self.omega2 * alpha)

        if self.verbose:  ## DEBUG
            print("## WARNING: Armijo reached max trials")
        return alpha

    def solve(self, x0):
        """Main optimization loop (Algorithm 1)
        
        Args:
            x0 (np.ndarray): Initial decision variables
            
        Returns:
            tuple: (x_opt, history) where:
                x_opt: Final solution approximation
                history: List of all iterates
        """
        print(f"\n## DEBUG: Starting optimization with initial point: {x0}")  ## DEBUG
        x = np.array(x0, dtype=float)
        history = [x.copy()]
        
        for k in range(self.max_iter):
            print(f"\n## DEBUG: Iteration {k+1}/{self.max_iter}")  ## DEBUG
            print(f"Current x: {x}")  ## DEBUG
            
            # Solve subproblem
            try:
                p, theta = self.solve_subproblem(x)
                d = p - x
                print(f"## DEBUG: Direction vector: {d}")  ## DEBUG
                print(f"## DEBUG: Gap value θ: {theta:.4e}")  ## DEBUG
                
                # Check convergence
                if abs(theta) < self.tol:
                    print(f"\n## DEBUG: Converged at iteration {k+1}")  ## DEBUG
                    break
                    
                # Line search
                alpha = self.armijo_line_search(x, d, theta)
                print(f"## DEBUG: Selected step size: {alpha:.4e}")  ## DEBUG
                
                # Update
                x_new = x + alpha * d
                f_new = [f(x_new) for f, _ in self.problem.f_list()]
                print(f"## DEBUG: New objectives: {f_new}")  ## DEBUG
                
                x = x_new
                history.append(x.copy())
                
            except Exception as e:
                print(f"\n## ERROR: Failed at iteration {k+1}")
                print(f"Current x: {x}")
                print(f"Error: {str(e)}")
                break

        else:  ## DEBUG
            print(f"\n## DEBUG: Reached maximum iterations ({self.max_iter})")

        print(f"\n## DEBUG: Optimization completed")  ## DEBUG
        print(f"Final x: {x}")  ## DEBUG
        return x, np.array(history)

# Example usage with debugging enabled
if __name__ == "__main__":
    problem = JOS1(n=2, bounds=[(-100.0, 100.0)]*2)
    solver = FrankWolfeMOP(problem, 
                         max_iter=100, 
                         tol=1e-6,
                         verbose=True)  ## DEBUG: Enable verbose output
    
    x0 = np.array([0.0, 0.0])
    print(f"Initial objectives: {problem.f1(x0):.4f}, {problem.f2(x0):.4f}")  ## DEBUG
    
    x_opt, history = solver.solve(x0)
    
    print("\nFinal results:")
    print(f"Optimal solution: {x_opt}")
    print(f"Final objectives: {problem.f1(x_opt):.4f}, {problem.f2(x_opt):.4f}")
    print(f"Path length: {len(history)} iterations")