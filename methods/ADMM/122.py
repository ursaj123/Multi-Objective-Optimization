# import numpy as np
# from scipy.optimize import minimize_scalar

# # --------------------------------
# # Problem 1: Convex Pareto Front
# # F = [x², (x-2)²]
# # --------------------------------

# def x_update_problem1(d, x, z, u, rho, ell):
#     term1 = 2 * x * d + (rho/2) * (x + d - z + u)**2
#     term2 = (rho/2) * (x + d - z + u)**2  # For f2 (zero gradient)
#     return max(term1, term2) + (ell/2)*d**2

# def z_update_problem1(d, x_new, z, u, rho, ell):
#     term1 = (rho/2) * (x_new - (z + d) + u)**2  # For g1 (zero)
#     term2 = (z + d - 2)**2 + (rho/2) * (x_new - (z + d) + u)**2
#     return max(term1, term2) + (ell/2)*d**2

# def solve_problem1(rho=1.0, ell=1.0, max_iter=100, tol=1e-6):
#     x, z, u = np.random.rand(), np.random.rand(), np.random.rand()  # Initialization
#     history = []
    
#     for _ in range(max_iter):
#         # X-update
#         res = minimize_scalar(lambda d: x_update_problem1(d, x, z, u, rho, ell))
#         x_new = x + res.x
        
#         # Z-update
#         res = minimize_scalar(lambda d: z_update_problem1(d, x_new, z, u, rho, ell))
#         z_new = z + res.x
        
#         # Dual update
#         u += x_new - z_new
        
#         # Check convergence
#         if abs(x_new - z_new) < tol and abs(z_new - z) < tol:
#             break
            
#         x, z = x_new, z_new
#         history.append((x, z))
    
#     return x, z, history

# # --------------------------------
# # Problem 2: Concave Pareto Front
# # F = [x, 1 - x²]
# # --------------------------------

# def x_update_problem2(d, x, z, u, rho, ell):
#     term1 = 1 * d + (rho/2) * (x + d - z + u)**2  # ∇f1 = 1
#     term2 = (rho/2) * (x + d - z + u)**2           # ∇f2 = 0
#     return max(term1, term2) + (ell/2)*d**2

# def z_update_problem2(d, x_new, z, u, rho, ell):
#     term1 = (rho/2) * (x_new - (z + d) + u)**2     # g1 = 0
#     term2 = (1 - (z + d)**2) + (rho/2)*(x_new - (z + d) + u)**2
#     return max(term1, term2) + (ell/2)*d**2

# def solve_problem2(rho=1.0, ell=1.0, max_iter=100, tol=1e-6):
#     x, z, u = np.random.rand(), np.random.rand(), np.random.rand()  # Initialization
#     history = []
    
#     for _ in range(max_iter):
#         # X-update
#         res = minimize_scalar(lambda d: x_update_problem2(d, x, z, u, rho, ell))
#         x_new = x + res.x
        
#         # Z-update
#         res = minimize_scalar(lambda d: z_update_problem2(d, x_new, z, u, rho, ell))
#         z_new = z + res.x
        
#         # Dual update
#         u += x_new - z_new
        
#         # Check convergence
#         if abs(x_new - z_new) < tol and abs(z_new - z) < tol:
#             break
            
#         x, z = x_new, z_new
#         history.append((x, z))
    
#     return x, z, history

# # --------------------------------
# # Run both problems
# # --------------------------------
# if __name__ == "__main__":
#     # Solve Problem 1 (Convex front)
#     x1, z1, _ = solve_problem1()
#     print(f"Problem 1 solution: x = {x1:.4f}, z = {z1:.4f}")
#     print(f"Objectives: f1 = {x1**2:.4f}, f2 = {(x1-2)**2:.4f}\n")
    
#     # Solve Problem 2 (Concave front)
#     x2, z2, _ = solve_problem2()
#     print(f"Problem 2 solution: x = {x2:.4f}, z = {z2:.4f}")
#     print(f"Objectives: f1 = {x2:.4f}, f2 = {1 - x2**2:.4f}")


# import numpy as np
# from scipy.optimize import minimize_scalar

# # --------------------------------
# # Modified MO-ADMM with Exploration
# # --------------------------------

# def solve_moadmm(rho=1.0, ell=1.0, max_iter=100, tol=1e-6, problem=1, exploration=0.1):
#     # Random initialization with exploration
#     np.random.seed()  # Ensure different seeds across runs
#     x = np.random.uniform(-2, 2)
#     z = np.random.uniform(-2, 2)
#     u = np.random.uniform(-1, 1)
    
#     history = []
    
#     for _ in range(max_iter):
#         # --- X-update with stochastic exploration ---
#         if problem == 1:
#             # For F = [x², (x-2)²]
#             def x_obj(d):
#                 perturb = exploration * np.random.normal()  # Add noise
#                 term1 = 2*(x + perturb)*d + (rho/2)*(x + d - z + u)**2
#                 term2 = (rho/2)*(x + d - z + u)**2  # For f2
#                 return max(term1, term2) + (ell/2)*d**2
#         else:
#             # For F = [x, 1 - x²]
#             def x_obj(d):
#                 perturb = exploration * np.random.normal()
#                 term1 = (1 + perturb)*d + (rho/2)*(x + d - z + u)**2
#                 term2 = (rho/2)*(x + d - z + u)**2
#                 return max(term1, term2) + (ell/2)*d**2
        
#         res = minimize_scalar(x_obj)
#         x_new = x + res.x
        
#         # --- Z-update with adaptive regularization ---
#         if problem == 1:
#             def z_obj(d):
#                 term1 = (rho/2)*(x_new - (z + d) + u)**2
#                 term2 = (z + d - 2)**2 + (rho/2)*(x_new - (z + d) + u)**2
#                 return max(term1, term2) + (ell/2 + 0.1*np.random.rand())*d**2  # Vary ell
#         else:
#             def z_obj(d):
#                 term1 = (rho/2)*(x_new - (z + d) + u)**2
#                 term2 = (1 - (z + d)**2) + (rho/2)*(x_new - (z + d) + u)**2
#                 return max(term1, term2) + (ell/2 + 0.1*np.random.rand())*d**2
        
#         res = minimize_scalar(z_obj)
#         z_new = z + res.x
        
#         # Dual update
#         u += x_new - z_new + exploration * np.random.normal()  # Perturb dual
        
#         if abs(x_new - z_new) < tol:
#             break
            
#         x, z = x_new, z_new
#         history.append((x, z))
    
#     return x, z, history

# # --------------------------------
# # Test Cases with Exploration
# # --------------------------------
# if __name__ == "__main__":
#     # Run 5 trials for each problem
#     for problem in [1, 2]:
#         print(f"\n{'Convex' if problem==1 else 'Concave'} Front Solutions:")
#         for _ in range(5):
#             x, z, _ = solve_moadmm(problem=problem, exploration=0.2)
#             if problem == 1:
#                 print(f"x={x:.3f}, f1={x**2:.3f}, f2={(x-2)**2:.3f}")
#             else:
#                 print(f"x={x:.3f}, f1={x:.3f}, f2={1 - x**2:.3f}")


import numpy as np
from scipy.optimize import minimize_scalar

# --------------------------------
# Corrected MO-ADMM with Projection
# --------------------------------

def solve_moadmm(rho=1.0, ell=1.0, max_iter=100, tol=1e-6, problem=1):
    np.random.seed()
    x = np.random.uniform(-2, 2)
    z = np.random.uniform(-2, 2)
    u = np.random.uniform(-1, 1)
    history = []
    
    for _ in range(max_iter):
        # --- X-update ---
        if problem == 1:
            # Convex problem: F = [x², (x-2)²]
            def x_obj(d):
                term1 = 2*(x)*d + (rho/2)*(x + d - z + u)**2
                term2 = (rho/2)*(x + d - z + u)**2  # For f2
                return max(term1, term2) + (ell/2)*d**2
        else:
            # Concave problem with CONSTRAINT: F = [x, 1-x²], x ∈ [-1,1]
            def x_obj(d):
                term1 = 1*d + (rho/2)*(x + d - z + u)**2  # ∇f1 = 1
                term2 = -2*(x)*d + (rho/2)*(x + d - z + u)**2  # ∇f2 = -2x
                return max(term1, term2) + (ell/2)*d**2
        
        res = minimize_scalar(x_obj)
        x_new = x + res.x
        
        # --- Z-update with projection for problem 2 ---
        if problem == 1:
            # Unconstrained z for convex problem
            def z_obj(d):
                term1 = (rho/2)*(x_new - (z + d) + u)**2  # g1 = 0
                term2 = (z + d - 2)**2 + (rho/2)*(x_new - (z + d) + u)**2
                return max(term1, term2) + (ell/2)*d**2
        else:
            # Project z to [-1,1] for concave problem
            z_candidate = x_new + u
            z_new = np.clip(z_candidate, -1, 1)  # Projection
        
        if problem == 1:
            res = minimize_scalar(z_obj)
            z_new = z + res.x
        else:
            # Direct projection for constrained case
            pass  
        
        # Dual update
        u += x_new - z_new
        
        if abs(x_new - z_new) < tol:
            break
            
        x, z = x_new, z_new
        history.append((x, z))
    
    return x, z, history

# --------------------------------
# Test Cases with Corrections
# --------------------------------
if __name__ == "__main__":
    print("Convex Front Solutions:")
    for _ in range(50):
        x, z, _ = solve_moadmm(problem=1)
        print(f"x={x:.3f}, f1={x**2:.3f}, f2={(x-2)**2:.3f}")
    
    print("\nConcave Front Solutions:")
    for _ in range(5):
        x, z, _ = solve_moadmm(problem=2)
        print(f"x={x:.3f}, f1={x:.3f}, f2={1 - x**2:.3f} (z={z:.3f})")