import numpy as np
from scipy.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import matplotlib.pyplot as plt
import os
import sys
from matplotlib import style
style.use('ggplot')

def zdt2(x):
    """ZDT2 Problem Definition"""
    # n = len(x)
    # f1 = x[0]
    # g = 1 + 9/(n-1)*np.sum(x[1:])
    # h = 1 - (f1/g)**2
    # f2 = g * h
    f1 = x
    f2 = 1-x**2
    return np.array([f1, f2])

def admm_zdt2(n_var=1, rho=1.0, max_iter=100, n_runs=20):
    """ADMM-based Pareto Front Generation for ZDT2"""
    bounds = [(0, 1)] * n_var
    pareto_points = []

    for _ in range(n_runs):
        # Initialize variables
        x = np.random.rand(n_var)
        z = np.copy(x)
        u1 = np.random.rand(n_var)
        u2 = np.random.rand(n_var)
        alpha = 0.0

        for _ in range(max_iter):
            # x-update: Solve min-max problem with regularization
            def obj_func(vars):
                x, t = vars[:-1], vars[-1]
                return t

            constraints = [
                {'type': 'ineq', 'fun': lambda v: v[-1] - (v[0] + 
                                 rho/2*np.sum((v[:-1]-z+u1)**2) +
                                 alpha*np.sum((v[:-1]-x)**2))},
                {'type': 'ineq', 'fun': lambda v: v[-1] - (zdt2(v[:-1])[1] + 
                                 rho/2*np.sum((v[:-1]-z+u2)**2) +
                                 alpha*np.sum((v[:-1]-x)**2))}
            ]

            res = minimize(obj_func, 
                         x0=np.concatenate([x, [0]]), 
                         constraints=constraints,
                         bounds=bounds + [(None, None)])
            x_new = res.x[:-1]

            # z-update: Consensus averaging
            z_new = 0.5 * (x_new + u1 + x_new + u2)

            # Dual updates
            u1 += 0.7 * (x_new - z_new)
            u2 += 0.7 * (x_new - z_new)

            x = x_new
            z = z_new
            alpha *= 0.95  # Reduce regularization

            pareto_points.append(zdt2(x))

    # Filter non-dominated solutions
    pareto_points = np.unique(pareto_points, axis=0)
    front = pareto_points[NonDominatedSorting().do(pareto_points, only_non_dominated_front=True)]
    return front[np.argsort(front[:, 0])]

# Generate and display results
pf4 = admm_zdt2(n_var=5)
print("ZDT2 Pareto Front:")
print(pf4)
print("\nProblem 4 Pareto Front:")
plt.figure(figsize=(5, 5))
plt.scatter(pf4[:, 0], pf4[:, 1], color='green')
plt.title('Problem 4 Pareto Front')
plt.xlabel('f1(x)')
plt.ylabel('f2(x)')
# plt.grid(True)
# plt.show()
# plt.close()
os.makedirs(os.path.join(os.path.dirname(__file__), 'results'), exist_ok=True)
plt.savefig(os.path.join(os.path.dirname(__file__), 'results', 'ZDT_pareto_front.png'))
print(pf4)