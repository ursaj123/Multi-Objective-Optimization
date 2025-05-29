import numpy as np
from scipy.optimize import minimize, Bounds
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

import os 
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../..", "problems/problem_lists"))

from ZDT2 import ZDT2
from JOS1 import JOS1
from DUMMY1 import DUMMY1
from DUMMY2 import DUMMY2

# print("Modules imported successfully")
# sys.exit()

def run_admm_problem1(rho=1.0, max_iter=100):
    """ADMM for Problem 1 with customizable dual initialization"""
    # x, z = 1.0, 1.0  # Neutral initialization
    x, z = np.random.uniform(-5, 5), np.random.uniform(-5, 5)  # Random initialization
    # u1, u2 = u_init
    # u1, u2 = np.random.rand(), np.random.rand()  # Random initialization  
    u1, u2 = np.random.uniform(-5, 5), np.random.uniform(-5, 5)  # Random initialization
    history = []
    prim_res, dual_res = [], []
    
    for _ in range(max_iter):
        # x-update: min max [x² + ρ/2(x-z+u1)², (x-2)² + ρ/2(x-z+u2)²]
        cons = [
            {'type': 'ineq', 'fun': lambda x: -x[0]**2 - rho/2*(x[0] - z + u1)**2 + x[1]},
            {'type': 'ineq', 'fun': lambda x: -(x[0]-2)**2 - rho/2*(x[0] - z + u2)**2 + x[1]}
        ]
        res = minimize(lambda x: x[1], [x, 0], constraints=cons, bounds=[(-5, 5), (5, None)])
        # res = minimize(lambda x: x[1], [x, 0], constraints=cons, bounds=Bounds([-5, -np.inf], [5, np.inf]))
        # res = minimize(lambda x: x[1], [x, 0], constraints=cons)
        x_new = res.x[0]

        # z-update (closed-form solution)
        z_new = (rho*(x_new + u1) + rho*(x_new + u2)) / (2*rho)
        
        # Dual updates
        u1 += x_new - z_new
        u2 += x_new - z_new
        
        dual_res.append(abs(z_new - z))

        x, z = x_new, z_new

        history.append((x**2, (x-2)**2))
        prim_res.append(abs(x_new - z_new))
        
    
    return np.array(history[-1]), np.array(prim_res[-1]), np.array(dual_res[-1])

def run_admm_problem2(rho=1.0, max_iter=100):
    """ADMM for Problem 2 with customizable dual initialization"""
    x, z = np.random.rand(), np.random.rand()  # Random initialization
    u1, u2 = np.random.rand(), np.random.rand()  # Random initialization
    history = []
    prim_res, dual_res = [], []
    
    for _ in range(max_iter):
        # x-update: min max [x + ρ/2(x-z+u1)², 1-x² + ρ/2(x-z+u2)²]
        cons = [
            {'type': 'ineq', 'fun': lambda x: -x[0] - rho/2*(x[0] - z + u1)**2 + x[1]},
            {'type': 'ineq', 'fun': lambda x: -(1-x[0]**2) - rho/2*(x[0] - z + u2)**2 + x[1]}
        ]
        # res = minimize(lambda x: x[1], [x, 0], constraints=cons, bounds=[(-5, None), (5, None)])
        res = minimize(lambda x: x[1], [x, 0], constraints=cons, bounds=[(-5, 5), (5, None)])
        # res = minimize(lambda x: x[1], [x, 0], constraints=cons)
        x_new = res.x[0]

        # z-update (closed-form solution)
        z_new = (rho*(x_new + u1) + rho*(x_new + u2)) / (2*rho)
        
        # Dual updates
        u1 += x_new - z_new
        u2 += x_new - z_new

        dual_res.append(abs(z_new - z))
        x, z = x_new, z_new
        history.append((x, 1 - x**2))
        prim_res.append(abs(x_new - z_new))
    
    return np.array(history[-1]), np.array(prim_res[-1]), np.array(dual_res[-1])


def run_admm_problem3(rho=1.0, max_iter=100):
    """ADMM for Problem 3 with customizable dual initialization DGO"""
    x, z = np.random.rand(), np.random.rand()  # Random initialization
    u1, u2 = np.random.rand(), np.random.rand()  # Random initialization
    history = []
    prim_res, dual_res = [], []
    
    for _ in range(max_iter):
        # x-update: min max [x + ρ/2(x-z+u1)², 1-x² + ρ/2(x-z+u2)²]
        cons = [
            {'type': 'ineq', 'fun': lambda x: -np.sin(x[0]) - rho/2*(x[0] - z + u1)**2 + x[1]},
            {'type': 'ineq', 'fun': lambda x: - np.sin(x[0]+0.7) - rho/2*(x[0] - z + u2)**2 + x[1]}
        ]
        # res = minimize(lambda x: x[1], [x, 0], constraints=cons, bounds=[(-10, 13), (None, None)])
        res = minimize(lambda x: x[1], [x, 0], constraints=cons)
        x_new = res.x[0]

        # z-update (closed-form solution)
        z_new = (rho*(x_new + u1) + rho*(x_new + u2)) / (2*rho)
        
        # Dual updates
        u1 += x_new - z_new
        u2 += x_new - z_new

        dual_res.append(abs(z_new - z))
        x, z = x_new, z_new
        history.append((np.sin(x), np.sin(x + 0.7)))
        prim_res.append(abs(x_new - z_new))
    
    return np.array(history[-1]), np.array(prim_res[-1]), np.array(dual_res[-1])



def run_admm_problem4(rho=1.0, max_iter=100):
    """ADMM for Problem 1 with customizable dual initialization"""
    # x, z = 1.0, 1.0  # Neutral initialization
    x, z = np.random.uniform(-5, 5, 2), np.random.uniform(-5, 5, 2)  # Random initialization
    # u1, u2 = u_init
    # u1, u2 = np.random.rand(), np.random.rand()  # Random initialization  
    u1, u2 = np.random.uniform(-5, 5, 2), np.random.uniform(-5, 5, 2)  # Random initialization
    history = []
    prim_res, dual_res = [], []
    
    for _ in range(max_iter):
        # x-update: min max [x² + ρ/2(x-z+u1)², (x-2)² + ρ/2(x-z+u2)²]
        cons = [
            {'type': 'ineq', 'fun': lambda x: -x[0]**2 - x[1]**2 - rho/2*(x[:2] - z + u1)**2 + x[2]},
            {'type': 'ineq', 'fun': lambda x: -(x[0]-2)**2 - (x[1]-2)**2 - rho/2*(x[:2] - z + u2)**2 + x[2]}
        ]
        # res = minimize(lambda x: x[2], np.concatenate([x, np.array([0])]), constraints=cons)
        res = minimize(lambda x: x[2], np.concatenate([x, np.array([0])]), constraints=cons, bounds=[(-5, 5), (-5, 5), (20, None)])
        x_new = res.x[:2]

        # z-update (closed-form solution)
        z_new = (rho*(x_new + u1) + rho*(x_new + u2)) / (2*rho)
        
        # Dual updates
        u1 += x_new - z_new
        u2 += x_new - z_new
        
        dual_res.append(abs(z_new - z))

        x, z = x_new, z_new

        history.append((x[0]**2 + x[1]**2, (x[0]-2)**2 + (x[1]-2)**2))
        prim_res.append(abs(x_new - z_new))
        
    
    return np.array(history[-1]), np.array(prim_res[-1]), np.array(dual_res[-1])

# def run_admm_problem4(rho=1.0, max_iter=100):
#     """ADMM for Problem 2 with customizable dual initialization ZDT2"""

#     n = 2
#     problem = JOS1(n=2, g_type=['zero'])
#     lb, ub = problem.lb, problem.ub
#     # print(f"lb: {lb}, ub: {ub}")
#     f_list = problem.f_list()
#     f1, f2 = f_list[0][0], f_list[1][0]

#     x, z = np.random.uniform(lb, ub, n), np.random.uniform(lb, ub, n)  # Random initialization
#     u1, u2 = np.random.uniform(lb, ub, n), np.random.uniform(lb, ub, n)  # Random initialization
#     history = []
#     prim_res, dual_res = [], []
    
#     for _ in range(max_iter):
#         # x-update: min max [x + ρ/2(x-z+u1)², 1-x² + ρ/2(x-z+u2)²]
#         cons = [
#             {'type': 'ineq', 'fun': lambda x: -f1(x[:n]) - rho/2*(x[:n] - z + u1)**2 + x[-1]},
#             {'type': 'ineq', 'fun': lambda x: - f2(x[:n]) - rho/2*(x[:n] - z + u2)**2 + x[-1]}
#         ]
#         # x0 = np.concatenate([x, np.array([0.0])])
#         # print(x0.shape)
#         res = minimize(lambda x: x[-1], x0 = np.concatenate([x, np.array([0.0])]), constraints=cons, bounds=problem.bounds + [(None, None)])
#         x_new = res.x[:n]

#         # z-update (closed-form solution)
#         z_new = (rho*(x_new + u1) + rho*(x_new + u2)) / (2*rho)
        
#         # Dual updates
#         u1 += x_new - z_new
#         u2 += x_new - z_new

#         dual_res.append(abs(z_new - z))
#         x, z = x_new, z_new
#         history.append((f1(x), f2(x)))
#         prim_res.append(abs(x_new - z_new))
    
#     return np.array(history[-1]), np.array(prim_res[-1]), np.array(dual_res[-1])

# ==================================================================
# Generate Pareto Fronts
# ==================================================================
def generate_pareto_front(problem_fn, n_points=100):
    """Generate non-dominated solutions through parameter variation"""
    solutions = []
    
    # Vary dual variable initialization to explore different tradeoffs
    for u_bias in np.linspace(-2, 2, n_points):
        sol, prim_res, dual_res = problem_fn()
        print(f"Solution: {sol}, Primal Residual: {prim_res}, Dual Residual: {dual_res}")
        solutions.append(sol)
    
    # Filter non-dominated solutions
    solutions = np.array(solutions)
    return solutions
    # print("Generated Solutions:", solutions)
    # nds = NonDominatedSorting().do(solutions, only_non_dominated_front=True)
    # return solutions[nds]

# ==================================================================
# Problem 1: f1(x) = x², f2(x) = (x-2)²
# ==================================================================
# pf1 = generate_pareto_front(run_admm_problem1)
# print("Problem 1 Pareto Front:")
# plt.figure(figsize=(5, 5))
# plt.scatter(pf1[:, 0], pf1[:, 1], color='blue')
# plt.title('Problem 1 Pareto Front')
# plt.xlabel('f1(x)')
# plt.ylabel('f2(x)')
# # plt.grid(True)
# # plt.show()
# # plt.close()
# plt.savefig('problem1_pareto_front.png')
# print(pf1)

# # ==================================================================
# # Problem 2: f1(x) = x, f2(x) = 1-x²
# # ==================================================================
# pf2 = generate_pareto_front(run_admm_problem2)
# print("\nProblem 2 Pareto Front:")
# plt.figure(figsize=(5, 5))
# plt.scatter(pf2[:, 0], pf2[:, 1], color='red')
# plt.title('Problem 2 Pareto Front')
# plt.xlabel('f1(x)')
# plt.ylabel('f2(x)')
# # plt.grid(True)
# # plt.show()
# # plt.close()
# plt.savefig('problem2_pareto_front.png')
# print(pf2)



# pf3 = generate_pareto_front(run_admm_problem3)
# print("\nProblem 3 Pareto Front:")
# plt.figure(figsize=(5, 5))
# plt.scatter(pf3[:, 0], pf3[:, 1], color='orange')
# plt.title('Problem 3 Pareto Front')
# plt.xlabel('f1(x)')
# plt.ylabel('f2(x)')
# # plt.grid(True)
# # plt.show()
# # plt.close()
# plt.savefig('problem3_pareto_front.png')
# print(pf3)



pf4 = generate_pareto_front(run_admm_problem4)
print("\nProblem 4 Pareto Front:")
plt.figure(figsize=(5, 5))
plt.scatter(pf4[:, 0], pf4[:, 1], color='green')
plt.title('Problem 4 Pareto Front')
plt.xlabel('f1(x)')
plt.ylabel('f2(x)')
# plt.grid(True)
# plt.show()
# plt.close()
plt.savefig('problem4_pareto_front.png')
# os.makedirs(os.path.join(os.path.dirname(__file__), 'results'), exist_ok=True)
# plt.savefig(os.path.join(os.path.dirname(__file__), 'results', 'JOS1_pareto_front.png'))
print(pf4)


