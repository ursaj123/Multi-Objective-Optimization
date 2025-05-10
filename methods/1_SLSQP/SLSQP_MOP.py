import numpy as np
from scipy.optimize import minimize
import sys
import os
sys.path.append('../problems')
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

class JOS1:
    """A multi-objective optimization problem with two quadratic objectives.
    
    Problem Formulation:
        minimize [f₁(x), f₂(x)]
        where:
            f₁(x) = 0.5 * Σxᵢ²    (Convex objective)
            f₂(x) = 0.5 * Σ(xᵢ - 2)²  (Shifted convex objective)
            
    Attributes:
        m (int): Number of objectives (fixed at 2)
        n (int): Number of decision variables
        bounds (list): Variable bounds [(-5, 5)]*n
    """
    def __init__(self, m=2, n=2):
        self.m = m
        self.n = n
        self.bounds = [(-5.0, 5.0) for _ in range(n)]

    def f1(self, x):
        """Calculate first objective value.
        
        Args:
            x (np.ndarray): Decision vector
            
        Returns:
            float: f₁(x) = 0.5 * ||x||²
        """
        return 0.5 * np.sum(x**2)

    def grad_f1(self, x):
        """Gradient of first objective.
        
        Args:
            x (np.ndarray): Decision vector
            
        Returns:
            np.ndarray: ∇f₁(x) = x
        """
        return x.copy()

    def f2(self, x):
        """Calculate second objective value.
        
        Args:
            x (np.ndarray): Decision vector
            
        Returns:
            float: f₂(x) = 0.5 * ||x - 2||²
        """
        return 0.5 * np.sum((x - 2)**2)

    def grad_f2(self, x):
        """Gradient of second objective.
        
        Args:
            x (np.ndarray): Decision vector
            
        Returns:
            np.ndarray: ∇f₂(x) = x - 2
        """
        return x - 2

    def f_list(self):
        return [
            (self.f1, self.grad_f1),
            (self.f2, self.grad_f2)
        ]

class SLSQP_MO:
    """Multi-objective SLSQP solver implementing the paper's algorithm.
    
    Algorithm Stages:
        1. Spread Stage: Expand Pareto front approximation using single-objective SQP steps
        2. Optimality Stage: Refine solutions using combined-objective SQP steps
    
    Key Formulations:
        - QP Subproblem for objective i (Spread Stage):
            min_d   ∇fᵢ(x)ᵀd + 0.5dᵀHᵢd
            s.t.    Linearized constraints (none in JOS1 case)
            
        - Combined QP (Optimality Stage):
            min_v   Σ[∇fᵢ(x)ᵀv + 0.5vᵀHᵢv]
            s.t.    ∇fᵢ(x)ᵀv ≤ 0 ∀i  (Non-deterioration constraints)
    """
    def __init__(self, problem, max_spread_iter=20, beta=0.5, mu=0.1, tol=1e-5, debug=True):
        self.problem = problem
        self.max_spread_iter = max_spread_iter
        self.beta = beta
        self.mu = mu
        self.tol = tol
        self.m = problem.m
        self.n = problem.n
        self.debug = debug

    def log(self, message):
        if self.debug:
            print(message)

    def solve(self, X0):
        """Main solver routine implementing the two-stage algorithm.
        
        Args:
            X0 (list): Initial population of points
            
        Returns:
            list: Approximated Pareto front
            
        Algorithm Flow:
            1. Spread Stage:
                - For each point and each objective:
                    a. Compute SQP direction
                    b. Perform Armijo line search
                    c. Maintain non-dominated set
                    
            2. Optimality Stage:
                - For each point:
                    a. Solve combined QP with non-deterioration constraints
                    b. Line search on combined objectives
                    c. Check convergence using point positions
        """
        Xk = self.non_dominated_sort(X0)
        self.log(f"\n=== Initial Non-Dominated Points ({len(Xk)}) ===")
        self.print_points(Xk)
        
        # Spread Stage
        for spread_iter in range(self.max_spread_iter):
            self.log(f"\n=== Spread Stage Iteration {spread_iter+1}/{self.max_spread_iter} ===")
            T = []
            for p_idx, x in enumerate(Xk):
                self.log(f"\nProcessing point {p_idx+1}/{len(Xk)}: {x}")
                
                for i, (fi, gradfi) in enumerate(self.problem.f_list()):
                    self.log(f"\n  Objective {i+1}:")
                    grad = gradfi(x)
                    H_i = np.eye(self.n)
                    d = -np.linalg.inv(H_i) @ grad
                    self.log(f"  Gradient: {grad}")
                    self.log(f"  Direction: {d} (norm: {np.linalg.norm(d):.2e})")
                    
                    if np.linalg.norm(d) < self.tol:
                        self.log("  Skipping - small direction")
                        continue
                        
                    # Line search
                    alpha = 1.0
                    f_curr = fi(x)
                    grad_d = grad @ d
                    self.log(f"  Initial alpha: {alpha:.2e}, f_curr: {f_curr:.2e}")
                    
                    while True:
                        x_new = x + alpha * d
                        f_new = fi(x_new)
                        if f_new <= f_curr + self.mu * alpha * grad_d:
                            break
                        alpha *= self.beta
                        if alpha < self.tol:
                            break
                            
                    self.log(f"  Final alpha: {alpha:.2e}")
                    if alpha >= self.tol:
                        T.append(x_new)
                        self.log(f"  Added new point: {x_new}")
                    else:
                        self.log("  No valid step found")
            
            if T:
                self.log(f"\nAdded {len(T)} new candidate points")
                Xk = self.non_dominated_sort(Xk + T)
                self.log(f"Current non-dominated points ({len(Xk)}):")
                self.print_points(Xk)
            else:
                self.log("No new points added in this iteration")
        
        # Optimality Stage
        self.log("\n=== Entering Optimality Stage ===")
        improved = True
        opt_iter = 0
        while improved:
            opt_iter += 1
            improved = False
            new_Xk = []
            self.log(f"\n--- Optimality Iteration {opt_iter} ---")
            
            for p_idx, x in enumerate(Xk):
                self.log(f"\nProcessing point {p_idx+1}/{len(Xk)}: {x}")
                grads = [gradfi(x) for (_, gradfi) in self.problem.f_list()]
                
                # Solve combined QP
                def objective(v):
                    return sum(g @ v + 0.5 * v @ v for g in grads)
                
                constraints = [{'type': 'ineq', 'fun': lambda v, g=g: -g @ v} for g in grads]
                res = minimize(objective, np.zeros(self.n), method='SLSQP', constraints=constraints)
                
                if not res.success:
                    self.log("  QP solve failed")
                    new_Xk.append(x)
                    continue
                
                v = res.x
                v_norm = np.linalg.norm(v)
                self.log(f"  QP solution found (v norm: {v_norm:.2e})")
                
                if v_norm < self.tol:
                    self.log("  Skipping - small direction")
                    new_Xk.append(x)
                    continue
                
                # Line search
                sum_f_curr = sum(fi(x) for (fi, _) in self.problem.f_list())
                sum_grad_v = sum(g @ v for g in grads)
                alpha = 1.0
                self.log(f"  Line search: sum_f_curr={sum_f_curr:.2e}, sum_grad_v={sum_grad_v:.2e}")
                
                while True:
                    x_new = x + alpha * v
                    sum_f_new = sum(fi(x_new) for (fi, _) in self.problem.f_list())
                    if sum_f_new <= sum_f_curr + self.mu * alpha * sum_grad_v:
                        break
                    alpha *= self.beta
                    if alpha < self.tol:
                        break
                
                self.log(f"  Final alpha: {alpha:.2e}")
                if alpha >= self.tol:
                    new_Xk.append(x_new)
                    improved = True
                    self.log(f"  Moved to new point: {x_new}")
                    self.log(f"  New objectives: {[fi(x_new) for (fi, _) in self.problem.f_list()]}")
                else:
                    new_Xk.append(x)
                    self.log("  No improvement")
            
            # Update Xk with non-dominated points
            filtered_Xk = self.non_dominated_sort(new_Xk)
            self.log("\nAfter optimality iteration:")
            self.print_points(filtered_Xk)
            
            # In the optimality stage loop:
            if len(filtered_Xk) == len(Xk):
                # Convert arrays to tuples for safe comparison
                sorted_old = sorted(Xk, key=lambda x: tuple(x.round(4)))
                sorted_new = sorted(filtered_Xk, key=lambda x: tuple(x.round(4)))
                if all(np.allclose(a, b) for a, b in zip(sorted_old, sorted_new)):
                    self.log("No changes detected, stopping optimality stage")
                    break
            Xk = filtered_Xk
        
        return Xk

    def non_dominated_sort(self, points):
        """Filter points to keep only non-dominated solutions.
        
        Dominance Criteria:
            x dominates y iff:
                fᵢ(x) ≤ fᵢ(y) ∀i ∈ {1,..,m}
                ∃j | fⱼ(x) < fⱼ(y)
        
        Args:
            points (list): List of candidate points
            
        Returns:
            list: Non-dominated subset of points
        """
        if not points:
            return []
        obj_values = np.array([[fi(x) for (fi, _) in self.problem.f_list()] for x in points])
        non_dominated = []
        for i in range(len(obj_values)):
            dominated = False
            for j in range(len(obj_values)):
                if i == j:
                    continue
                if np.all(obj_values[j] <= obj_values[i]) and np.any(obj_values[j] < obj_values[i]):
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(points[i])
        return non_dominated

    def print_points(self, points):
        """Debug helper: Print points with objective values.
        
        Args:
            points (list): List of points to display
        """
        if not self.debug:
            return
        for i, x in enumerate(points):
            f_values = [f"{fi(x):.4f}" for (fi, _) in self.problem.f_list()]
            print(f"Point {i+1}: {x} => ({', '.join(f_values)})")

# Example usage
if __name__ == "__main__":
    problem = JOS1()
    # let us samplae 50 points of 2D array
    initial_X = [np.random.uniform(-5, 5, problem.n) for _ in range(5000)]
    # print(initial_X)
    # sys.exit()
    # initial_X = [np.random.uniform(-5, 5, problem.n) for _ in range(50)]
    # initial_X = [np.array([0.0, 0.0]), np.array([2.0, 2.0])]
    solver = SLSQP_MO(problem, max_spread_iter=2, debug=True)
    pareto_front = solver.solve(initial_X)
    

    f1_values, f2_values = [], []
    print("\n=== Final Pareto Front ===")
    for i, x in enumerate(pareto_front):
        f1 = problem.f1(x)
        f2 = problem.f2(x)
        f1_values.append(f1)
        f2_values.append(f2)
        print(f"Point {i+1}: {x} => f1: {f1:.4f}, f2: {f2:.4f}")


    # Plotting the Pareto front
    plt.figure(figsize=(8, 6))
    plt.scatter(f1_values, f2_values, c='blue', marker='o')
    plt.title('Pareto Front')
    plt.xlabel('Objective 1 (f1)')
    plt.ylabel('Objective 2 (f2)')
    plt.savefig('JOS1_SLSQP.png')


    