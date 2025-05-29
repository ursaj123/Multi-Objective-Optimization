import numpy as np
from scipy.optimize import minimize
from scipy.linalg import norm
import os, sys
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from time import time

def plot_3d_pareto_front(f_true=None, f_approx=None,
                          problem_name='',
                          elev=20, azim=-100,
                          color_true='black', color_approx='red',
                          label_true='True', label_approx='Approx',
                          point_size=5,
                          save_path=None):
    """
    Plots two 3D scatter plots in the same figure with custom orientation.

    Parameters:
    - f1_true, f2_true, f3_true: Arrays for the true values
    - f1_approx, f2_approx, f3_approx: Arrays for the approximate values
    - elev: Elevation angle for the view (default: 20)
    - azim: Azimuth angle for the view (default: -135)
    - color_true: Color of the true points (default: 'black')
    - color_approx: Color of the approximate points (default: 'red')
    - label_true: Label for the true points
    - label_approx: Label for the approximate points
    - point_size: Size of the scatter points (default: 5)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if f_true is not None:
        ax.scatter(f_true[:, 0], f_true[:, 1], f_true[:, 2], color=color_true, s=point_size, label=label_true)

    if f_approx is not None:
        ax.scatter(f_approx[:, 0], f_approx[:, 1], f_approx[:, 2], color=color_approx, s=point_size, label=label_approx)

    ax.set_xlabel(r'$f_1(x)$')
    ax.set_ylabel(r'$f_2(x)$')
    ax.set_zlabel(r'$f_3(x)$')

    ax.view_init(elev=elev, azim=azim)
    ax.legend()
    ax.set_title(f'Pareto Front Comparison {problem_name}')
    plt.tight_layout()
    plt.show()
    if save_path:
        plt.savefig(save_path)

    plt.close(fig)

class MultiObjectiveProblem:
    def __init__(self, f_list, grad_f_list, bounds, g_type):
        self.f_list = f_list              # List of objective functions
        self.grad_f_list = grad_f_list    # List of gradient functions
        self.bounds = bounds              # Bounds: [[a1, b1], ..., [an, bn]]
        self.g_type = g_type            # List of strings: ['zero', 'l1', ...]

    def evaluate_f(self, x):
        return np.array([f(x) for f in self.f_list])
    
    def evaluate_g(self, x):
        g_eval = np.zeros(len(self.f_list))
        if self.g_type=='zero':
            pass
        else:
            pass
        return g_eval
    
    def evaluate(self, x, z):
        """Evaluate all objectives at x"""
        f_eval = self.evaluate_f(x)
        g_eval = self.evaluate_g(z)
        return f_eval + g_eval
        

def initialize_x(bounds):
    """Initialize x within bounds"""
    return np.array([np.random.uniform(low, high) for (low, high) in bounds])

# print(initialize_x([(-5, 5), (-5, 5)]))  # Example usage)
# sys.exit()

def x_subproblem(grad_list, rho, d_z, lam, norm_constraint='L1'):
    """Solve x-direction subproblem: min max_i [∇f_i^T d_x] + (ρ/2)||d_x - d_z + λ/ρ||^2"""
    n = len(d_z)
    
    # Augmented Lagrangian term with scaled dual variable
    def objective(vars):
        d_x, alpha = vars[:n], vars[n]
        return alpha + (rho/2) * norm(d_x - d_z + lam/rho)**2
    
    # Constraints: ∇f_i^T d_x <= α (for all i), ||d_x|| <= 1
    constraints = [
        {'type': 'ineq', 'fun': lambda vars, i=i: vars[n] - np.dot(grad_list[i], vars[:n])}
        for i in range(len(grad_list))]

    if norm_constraint == 'L2':
        # L1 norm constraint: ||d_x||_2 <= 1
        constraints = constraints + [{'type': 'ineq', 'fun': lambda vars: 1 - norm(vars[:n])}]
        res = minimize(objective, x0=np.zeros(n+1), constraints=constraints, method='SLSQP')
        return res.x[:n], res.x[n]
    
    # L2 norm constraint: ||d_x||_1 <= 1
    res = minimize(objective, x0=np.zeros(n+1), constraints=constraints, method='SLSQP',
    bounds = [(-1, 1)]*n + [(None, None)])
    return res.x[:n], res.x[n]
    
    
    

def z_subproblem(d_x, lam, rho, g_type, norm_constraint='L1'):
    """Solve z-direction subproblem for a single g_type"""
    n = len(d_x)
    
    if g_type == 'zero':
        # For g=0, solve: min (ρ/2)||d_x - d_z + λ/ρ||^2 with ||d_z|| <= 1
        def objective(d_z):
            return (rho/2) * norm(d_x - d_z + lam/rho)**2
        if norm_constraint == 'L2':
            cons = {'type': 'ineq', 'fun': lambda d_z: 1 - norm(d_z)}
            res = minimize(objective, x0=np.zeros(n), constraints=cons, method='SLSQP')
            return res.x
        
        # L1 norm constraint: ||d_z||_1 <= 1
        res = minimize(objective, x0=np.zeros(n), method='SLSQP', bounds = [(-1, 1)]*n)
        return res.x

    else:
        pass

def subadmm_single_lambda(x, grad_list, g_type, rho, eps_inner, max_inner):
    """SubADMM inner loop for Algorithm 2a"""
    n = len(x)
    # d_x, d_z, lam = np.zeros(n), np.zeros(n), np.zeros(n)  # Initialize variables
    d_x, d_z, lam = np.random.rand(n), np.random.rand(n), np.random.rand(n)  # Initialize variables
    
    for p in range(max_inner):
        print(f'Inner iteration {p}')
        # X-direction update
        d_x_new, alpha = x_subproblem(grad_list, rho, d_z, lam)
        print(f"Inner alpha to check convergence: alpha = {alpha}")
        
        # Z-direction update (assume all g_i have the same type for simplicity)
        # For multi-type, loop over g_types and combine results
        # if not all(g == 'zero' for g in g_types):
        #     raise NotImplementedError("Mixed g_types not supported")

        d_z_new = d_z.copy()
        if g_type=='zero':
            d_z_new = z_subproblem(d_x_new, lam, rho, g_type)
        else:
            pass
        
        # Dual update: λ = λ + ρ(d_x - d_z)
        lam_new = lam + rho * (d_x_new - d_z_new)
        
        # Termination check
        primal_res = norm(d_x_new - d_z_new)
        dual_res = norm(d_z_new - d_z)

        if primal_res < eps_inner and dual_res < eps_inner:
            # print(f"Converged at iteration {p}: primal_res={primal_res}, dual_res={dual_res}")
            return d_x_new, d_z_new

        d_x, d_z, lam = d_x_new, d_z_new, lam_new
    
    return d_x, d_z

def algo_2a(mop, rho=1.0, beta=0.5, epsilon=1e-6, max_outer=100, eps_inner=1e-6, max_inner=50):
    """Main algorithm with single lambda"""
    n = len(mop.bounds)
    x = initialize_x(mop.bounds)
    z = x.copy()
    x_history, z_history = [x.copy()], [z.copy()]
    
    for k in range(max_outer):
        # Compute gradients at current x
        grad_list = [grad_f(x) for grad_f in mop.grad_f_list]
        
        # Solve SUBADMM for directions
        d_x, d_z = subadmm_single_lambda(x, grad_list, mop.g_type, rho, eps_inner, max_inner)
        
        # Line search with bounds check
        
        t = 1.0
        print('Line Search Starts')
        print(f"Line search: x = {x}, t={t}, d_x={d_x}, d_z={d_z}")
        for _ in range(10):
            x_new = x + t * d_x
            z_new = z + t * d_z
            # Check if within bounds
            in_bounds1 = all(low <= xi <= high for xi, (low, high) in zip(x_new, mop.bounds))
            in_bounds2 = all(low <= zi <= high for zi, (low, high) in zip(z_new, mop.bounds))
            if in_bounds1 and in_bounds2:
                # Check Armijo condition for all objectives
                eval_g_new = mop.evaluate_g(z_new)
                eval_g = mop.evaluate_g(z)
                armijo_ok = all(
                    mop.f_list[i](x_new) + eval_g_new[i] <= mop.f_list[i](x) + eval_g[i] + beta * t * np.dot(grad_list[i], d_x)
                    for i in range(len(mop.f_list))
                )
                if armijo_ok:
                    break
            t *= 0.5

        print(f"Line search: x_new = {x_new}, t={t}, d_x={d_x}, d_z={d_z}")
        print('Line Search Ends')
        
        # Update x and check termination
        x_prev = x.copy()
        z_prev = z.copy()
        x = x_new.copy()
        z = z_new.copy()
        x_history.append(x.copy())
        z_history.append(z.copy())
        if norm(x - x_prev) < epsilon:
            print(f"Converged at outer iteration: x={x}, z={z}, k = {k}, ||x-x_prev||={norm(x - x_prev)}, ||dx|| = {norm(d_x)}")
            return np.array(x_history), np.array(z_history)

        print(f"Outer iteration {k}: x={x}, z={z}, ||x-x_prev||={norm(x - x_prev)}, ||dx|| = {norm(d_x)}")
    
    print(f"Max outer iterations reached: {x}, z={z}, {max_outer} = k, ||x-z||={norm(x - z)}")
    return np.array(x_history), np.array(z_history)



import numpy as np
# from algo_2a import MultiObjectiveProblem as MOP2a, algo_2a
# from algo_2b import MultiObjectiveProblem as MOP2b, algo_2b

# Define a 2D problem with two objectives and g=0
# JOS1 problem
def f1(x): return 0.5*np.linalg.norm(x)**2
def f2(x): return 0.5*np.linalg.norm(x-1)**2
def grad_f1(x): return x
def grad_f2(x): return x-1
bounds = [(-5, 5)]*50
g_type = 'zero'  # Both objectives have g=0



# ZDT2 
# def f1(x):
#     # return x[0]
#     return x[0]

# def f2(x):
#     # return 1 - x[0]**2
#     n = len(x)
#     if n == 1:
#         g = 1.0
#     else:
#         g = 1 + 9 * np.sum(x[1:]) / (n - 1)
#     return g * (1 - (x[0] / g) ** 2)

# def grad_f1(x):
#     # return np.array([1.0])
#     grad = np.zeros_like(x)
#     grad[0] = 1
#     return grad
    

# def grad_f2(x):
#     # return np.array([-2 * x[0]])
#     n = len(x)
#     if n == 1:
#         g = 1.0
#     else:
#         g = 1 + 9 * np.sum(x[1:]) / (n - 1)
#     grad = np.zeros_like(x)
#     grad[0] = -2 * x[0] / g
#     if n > 1:
#         common_term = (9 / (n - 1)) * (1 - (x[0]/g)**2)
#         grad[1:] = common_term
#     return grad

# bounds = [(0, 1)]*2
# g_type = 'zero'  # Both objectives have g=0

# fonseca-fleming problems
# def f1(x):
#     n = len(x)
#     c = 1 / np.sqrt(n)
#     sum_term = np.sum((x - c) ** 2)
#     return 1 - np.exp(-sum_term)

# def f2(x):
#     n = len(x)
#     c = 1 / np.sqrt(n)
#     sum_term = np.sum((x + c) ** 2)
#     return 1 - np.exp(-sum_term)

# def grad_f1(x):
#     n = len(x)
#     c = 1 / np.sqrt(n)
#     sum_term = np.sum((x - c) ** 2)
#     grad = 2 * np.exp(-sum_term) * (x - c)
#     return grad

# def grad_f2(x):
#     n = len(x)
#     c = 1 / np.sqrt(n)
#     sum_term = np.sum((x + c) ** 2)
#     grad = 2 * np.exp(-sum_term) * (x + c)
#     return grad

# # Define bounds and g_type
# bounds = [(-4, 4)] * 2
# g_type = 'zero'  # Both objectives have g=0


# kursawe
# def f1(x):
#     n = len(x)
#     total = 0.0
#     for i in range(n - 1):
#         xi = x[i]
#         xj = x[i + 1]
#         total += -10 * np.exp(-0.2 * np.sqrt(xi**2 + xj**2))
#     return total

# def f2(x):
#     total = 0.0
#     for xi in x:
#         total += np.abs(xi)**0.8 + 5 * np.sin(xi)**3
#     return total

# def grad_f1(x):
#     n = len(x)
#     grad = np.zeros_like(x)
#     for i in range(n - 1):
#         xi = x[i]
#         xj = x[i + 1]
#         s_sq = xi**2 + xj**2
#         if s_sq == 0:
#             deriv_i = 0.0
#             deriv_j = 0.0
#         else:
#             s = np.sqrt(s_sq)
#             common = 2 * np.exp(-0.2 * s) / s
#             deriv_i = common * xi
#             deriv_j = common * xj
#         grad[i] += deriv_i
#         grad[i + 1] += deriv_j
#     return grad

# def grad_f2(x):
#     grad = np.zeros_like(x)
#     eps = 1e-12  # Small epsilon to avoid division by zero
#     for i in range(len(x)):
#         xi = x[i]
#         abs_xi = np.abs(xi)
#         sign = np.sign(xi)
#         # Handle derivative of |x_i|^0.8
#         d_abs = 0.8 * sign * (abs_xi + eps) ** (-0.2)
#         # Handle derivative of 5*sin(x_i)^3
#         sin_xi = np.sin(xi)
#         cos_xi = np.cos(xi)
#         d_sin = 15 * (sin_xi ** 2) * cos_xi
#         grad[i] = d_abs + d_sin
#     return grad

# bounds = [(-5, 5)] * 3
# g_type = 'zero'  # Both objectives have g=0


# poloni problem
# Precompute constants A1 and A2
# A1 = 0.5 * np.sin(1) - 2 * np.cos(1) + np.sin(2) - 1.5 * np.cos(2)
# A2 = 1.5 * np.sin(1) - np.cos(1) + 2 * np.sin(2) - 0.5 * np.cos(2)

# def f1(x):
#     x1, x2 = x
#     B1 = 0.5 * np.sin(x1) - 2 * np.cos(x1) + np.sin(x2) - 1.5 * np.cos(x2)
#     B2 = 1.5 * np.sin(x1) - np.cos(x1) + 2 * np.sin(x2) - 0.5 * np.cos(x2)
#     return 1 + (A1 - B1)**2 + (A2 - B2)**2

# def f2(x):
#     x1, x2 = x
#     return (x1 + 3)**2 + (x2 + 1)**2

# def grad_f1(x):
#     x1, x2 = x
#     B1 = 0.5 * np.sin(x1) - 2 * np.cos(x1) + np.sin(x2) - 1.5 * np.cos(x2)
#     B2 = 1.5 * np.sin(x1) - np.cos(x1) + 2 * np.sin(x2) - 0.5 * np.cos(x2)
    
#     dB1_dx1 = 0.5 * np.cos(x1) + 2 * np.sin(x1)
#     dB1_dx2 = np.cos(x2) + 1.5 * np.sin(x2)
#     dB2_dx1 = 1.5 * np.cos(x1) + np.sin(x1)
#     dB2_dx2 = 2 * np.cos(x2) + 0.5 * np.sin(x2)
    
#     term1 = A1 - B1
#     term2 = A2 - B2
    
#     df1_dx1 = -2 * term1 * dB1_dx1 - 2 * term2 * dB2_dx1
#     df1_dx2 = -2 * term1 * dB1_dx2 - 2 * term2 * dB2_dx2
    
#     return np.array([df1_dx1, df1_dx2])

# def grad_f2(x):
#     x1, x2 = x
#     df2_dx1 = 2 * (x1 + 3)
#     df2_dx2 = 2 * (x2 + 1)
#     return np.array([df2_dx1, df2_dx2])

# bounds = [(-np.pi, np.pi)] * 2
# g_type = 'zero'  # Both objectives have g=0


# vinnet

# def f1(x):
#     x_val, y_val = x[0], x[1]
#     return 0.5 * (x_val**2 + y_val**2) + np.sin(x_val**2 + y_val**2)

# def f2(x):
#     x_val, y_val = x[0], x[1]
#     return ((3*x_val - 2*y_val + 4)**2) / 8 + ((x_val - y_val + 1)**2) / 27 + 15

# def f3(x):
#     x_val, y_val = x[0], x[1]
#     return 1 / (x_val**2 + y_val**2 + 1) - 1.1 * np.exp(-x_val**2 - y_val**2)

# def grad_f1(x):
#     x_val, y_val = x[0], x[1]
#     common_term = 2 * np.cos(x_val**2 + y_val**2)
#     df_dx = x_val + x_val * common_term
#     df_dy = y_val + y_val * common_term
#     return np.array([df_dx, df_dy])

# def grad_f2(x):
#     x_val, y_val = x[0], x[1]
#     term1 = (3*x_val - 2*y_val + 4)
#     term2 = (x_val - y_val + 1)
    
#     # Corrected coefficients for term1 and term2
#     df_dx = (6 * term1) / 8 + (2 * term2) / 27  # (3/4)*term1 + (2/27)*term2
#     df_dy = (-4 * term1) / 8 - (2 * term2) / 27  # (-1/2)*term1 - (2/27)*term2
#     return np.array([df_dx, df_dy])

# def grad_f3(x):
#     x_val, y_val = x[0], x[1]
#     denominator = (x_val**2 + y_val**2 + 1) ** 2
#     exp_term = np.exp(-x_val**2 - y_val**2)
#     df_dx = (-2 * x_val) / denominator + 2.2 * x_val * exp_term
#     df_dy = (-2 * y_val) / denominator + 2.2 * y_val * exp_term
#     return np.array([df_dx, df_dy])

# # Define bounds and g_type
# bounds = [(-3, 3)]*2
# g_type = 'zero'  # Both objectives have g=0

# SD function
# def f1(x):
#     return 2*x[0] + np.sqrt(2)*x[1] + np.sqrt(2)*x[2] + x[3] 

# def f2(x):
#     return 2/x[0] + 2*np.sqrt(2)/x[1] + 2*np.sqrt(2)/x[2] + x[3] 

# def grad_f1(x):
#     grad = np.zeros_like(x)
#     grad[0] = 2
#     grad[1] = np.sqrt(2)
#     grad[2] = np.sqrt(2)
#     grad[3] = 1
#     return grad

# def grad_f2(x):
#     grad = np.zeros_like(x)
#     grad[0] = -2/(x[0]**2)
#     grad[1] = -2*np.sqrt(2)/(x[1]**2)
#     grad[2] = -2*np.sqrt(2)/(x[2]**2)
#     grad[3] = 1
#     return grad

# bounds = [(1, 3), (np.sqrt(2), 3), (np.sqrt(2), 3), (1, 3)]
# g_type = 'zero'  # Both objectives have g=0

# Initialize problem instances
start_time = time()
mop_2a = MultiObjectiveProblem(f_list=[f1, f2], grad_f_list=[grad_f1, grad_f2], bounds=bounds, g_type=g_type)
# mop_2a = MultiObjectiveProblem(f_list=[f1, f2, f3], grad_f_list=[grad_f1, grad_f2, grad_f3], bounds=bounds, g_type=g_type)

# mop_2b = MOP2b(f_list=[f1, f2], grad_f_list=[grad_f1, grad_f2], bounds=bounds, g_types=g_types)

# Run algorithms
history_2a, _ = algo_2a(mop_2a, rho=1.0, max_outer=10, max_inner=100)
# history_2b = algo_2b(mop_2b, rho=1.0, max_outer=100)

print("Algorithm 2a final point:", history_2a[-1])
# print("Algorithm 2b final point:", history_2b[-1])
# pareto_front = np.array([mop_2a.evaluate(p, p) for p in history_2a[-1:]])
print(f"Time taken for Algorithm 2a: {time() - start_time:.2f} seconds")

sys.exit()
# print(f"History 2a:\n", history_2a)


# sys.exit()
# Generate Pareto front with multiple runs
pareto_front = []
for _ in range(100):
    history, _ = algo_2a(mop_2a, rho=1.0, max_outer=500)
    print("Pareto point:", history[-1])
    pareto_front.append(history[-1])

pareto_front = np.array(pareto_front)
pareto_front = np.array([mop_2a.evaluate(p, p) for p in pareto_front])
# print("Pareto Front (10 points):\n", np.array(pareto_front))

# plot_3d_pareto_front(
#     f_approx=pareto_front,
#     problem_name='Vinnet',
#     save_path='expt5.png',
# )

# sys.exit()

plt.figure(figsize=(10, 6))
plt.scatter(pareto_front[:, 0], pareto_front[:, 1], c='blue', label='Pareto Front')
plt.title('Pareto Front for Multi-Objective Problem')
plt.xlabel('Objective 1')
plt.ylabel('Objective 2')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('expt5.png')

