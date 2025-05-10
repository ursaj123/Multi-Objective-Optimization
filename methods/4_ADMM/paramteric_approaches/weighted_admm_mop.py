from abc import abstractmethod
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from problems import *
import sys
# line number 79, 173,177

def soft_threshold(x, threshold):
    r"""Soft-thresholding operator for L1 norm (element-wise)."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)



def admm_multiobjective(f_list, g_types, weights, bounds = None, n=10, rho=1.0, max_iter=100, tol=1e-6, verbose=True):
    r"""
    Generalized ADMM solver for weighted multiobjective optimization.
    
    Args:
        f_list (list of tuples): Each tuple contains (f_i, grad_f_i) for the ith objective.
        g_types (list of str): 'l1' or 'none' for each g_i(z).
        weights (array): Weights for objectives (sum to 1).
        n (int): Dimension of the problem.
        rho (float): Penalty parameter.
        max_iter (int): Maximum iterations.
        tol (float): Convergence tolerance.
        verbose (bool): Print convergence messages.
        
    Returns:
        z (array): Consensus solution.
        history (dict): Primal/dual residuals per iteration.
    """
    m = len(weights)
    # assert len(f_list) == m and len(g_types) == m, "Mismatch in objectives or g_types."
    # assert np.isclose(np.sum(weights), "Weights must sum to 1."
    
    # Dimension from the first f_i's input shape (assume all x_i and z have same dimension)
    # n = len(f_list[0][0](np.zeros(1)))  # Test f_i with dummy input
    
    # Initialize variables
    z = np.random.randn(n)
    x = [np.random.randn(n) for _ in range(m)]
    y = [np.random.randn(n) for _ in range(m)]
    
    # Track residuals
    history = {'primal': [], 'dual': []}
    
    for it in range(max_iter):
        #-----------------------------------------------------------
        # 1. Update local variables x_i in parallel
        #-----------------------------------------------------------
        for i in range(m):
            # Define the x_i subproblem: min w_i f_i(x_i) + (rho/2)||x_i - z + y_i/rho||^2
            def obj_x_i(x_i):
                return weights[i] * f_list[i][0](x_i) + (rho/2) * np.sum((x_i - z + y[i]/rho)**2)
            
            def grad_x_i(x_i):
                # print(f_list[i][1](x_i))
                # print(weights[i])
                # print(weights[i]*(f_list[i][1](x_i)))
                return weights[i] * f_list[i][1](x_i) + rho * (x_i - z + y[i]/rho)
            
            res = minimize(obj_x_i, x[i], jac=grad_x_i, bounds=bounds, method='L-BFGS-B')
            x[i] = res.x
        
        #-----------------------------------------------------------
        # 2. Update global variable z
        #-----------------------------------------------------------
        # Compute average of (x_i + y_i/rho)
        v = [x[i] + y[i]/rho for i in range(m)]
        avg_v = np.mean(v, axis=0)
        
        # Check if any g_i is L1
        lambda_total = sum(weights[i] for i in range(m) if g_types[i] == 'l1')
        
        if lambda_total > 0:
            # Apply soft thresholding for L1 terms
            threshold = lambda_total / (rho * m)
            z_new = soft_threshold(avg_v, threshold)
            # z = np.clip(z, [bounds[i][0] for i in range(n)], [bounds[i][1] for i in range(n)])
        else:
            # Simple averaging if all g_i(z) = 0
            z_new = avg_v.copy()
        
        #-----------------------------------------------------------
        # 3. Update dual variables y_i
        #-----------------------------------------------------------
        for i in range(m):
            y[i] += rho * (x[i] - z_new)
        
        #-----------------------------------------------------------
        # Check convergence
        #-----------------------------------------------------------
        primal_res = np.sqrt(sum(np.linalg.norm(x[i] - z_new)**2 for i in range(m)))
        dual_res = rho * np.sqrt(m) * np.linalg.norm(z_new - z)
        
        history['primal'].append(primal_res)
        history['dual'].append(dual_res)
        
        if primal_res < tol and dual_res < tol:
            if verbose:
                print(f"Converged at iteration {it}")
            break
        
        z = z_new.copy()
    
    return z, history

#----------------------------------------------
# Example Usage with Two Test Problems
#----------------------------------------------
if __name__ == "__main__":
    m, n, num_samples = 2, 30, 100
    # m, n, num_samples = 1, 2, 2
    # problem = JOS1()
    # problem = Poloni()
    # problem = ZDT1()
    problem = ZDT2()
    # problem = Rosenbrock()
    z_hist_l1 = []
    z_hist_0 = []
    f_list = problem.f_list()
    bounds = problem.bounds
    
    print(f"Bounds: {bounds}")
    # print(f_list)
    # sys.exit()
    for i in range(num_samples):
        weights = np.random.rand(m)
        weights = weights / np.sum(weights)  # Normalize weights to sum to 1
        # print(f"Normalized weights: {weights}")
        
        # Test Problem 1: g_i(z) = L1
        z_opt1, _ = admm_multiobjective(
            f_list=f_list,
            g_types=['l1', 'l1'],  # Both objectives have L1 regularization
            weights=weights,
            bounds=bounds,
            n = n,
            rho=1.0,
            verbose=True
        )
        print(f"Test Problem 1 solution: z = {z_opt1}\n")  # Expected near 0
        z_hist_l1.append(z_opt1)
        
        # Test Problem 2: g_i(z) = 0
        z_opt2, _ = admm_multiobjective(
            f_list=f_list,
            g_types=['none', 'none'],  # No regularization
            weights=weights,
            bounds=bounds,
            n = n,
            rho=1.0,
            verbose=True
        )
        print(f"Test Problem 2 solution: z = {z_opt2}")  # Expected 1.0 (average of 0 and 2)
        z_hist_0.append(z_opt2)

    # print(f"z_hist_l1: {z_hist_l1}")
    # print(f"z_hist_0: {z_hist_0}")
    # sys.exit()
    # pareto front
    f1_values_l1, f2_values_l1 = [], []
    f1_values_0, f2_values_0 = [], []
    for z in z_hist_l1:
        # z = np.clip(z, [bounds[i][0] for i in range(n)], [bounds[i][1] for i in range(n)])
        f1_values_l1.append(f_list[0][0](z))
        f2_values_l1.append(f_list[1][0](z))
    for z in z_hist_0:
        # z = np.clip(z, [bounds[i][0] for i in range(n)], [bounds[i][1] for i in range(n)])
        f1_values_0.append(f_list[0][0](z))
        f2_values_0.append(f_list[1][0](z))
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(f1_values_l1, f2_values_l1, label='L1 Regularization', color='blue')
    plt.title('Pareto Front with L1 Regularization')
    plt.xlabel('f1(x)')
    plt.ylabel('f2(x)')
    plt.grid()
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.scatter(f1_values_0, f2_values_0, label='No Regularization', color='red')
    plt.title('Pareto Front with No Regularization')
    plt.xlabel('f1(x)')
    plt.ylabel('f2(x)')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Save the plot
    plt.savefig('pareto_fronts_ZDT2.png')

    