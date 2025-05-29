import numpy as np
from scipy.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

def is_pareto(costs):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True
    return is_efficient

def tchebychev_admm(objectives, bounds, utopia, n_weights=20, rho=10.0, max_iter=100):
    weights = np.array([[w, 1-w] for w in np.logspace(-3, 0, n_weights)])
    pareto_front = []
    
    for λ in weights:
        x = np.mean(bounds)
        z = x
        u = 0.0
        
        for _ in range(max_iter):
            # X-update with strict Tchebychev constraints
            def x_obj(var):
                x_val = var[0]
                alpha = var[1]
                f1, f2 = [obj(x_val) for obj in objectives]
                cons = [
                    λ[0]*(f1 - utopia[0]) - alpha,
                    λ[1]*(f2 - utopia[1]) - alpha
                ]
                penalty = 1e6 * sum(np.maximum(0, cons)**2)  # Strict enforcement
                return alpha + 0.5*rho*(x_val - z + u)**2 + penalty
            
            res = minimize(
                x_obj,
                [x, 0],
                bounds=[bounds, (None, None)],
                method='SLSQP',
                options={'ftol': 1e-8}
            )
            x_new = res.x[0]
            
            # Z-update with projection
            z_new = np.clip(x_new + u, *bounds)
            
            # Dual update
            u += x_new - z_new
            
            if abs(x_new - z_new) < 1e-8:
                break
            x, z = x_new, z_new
        
        pareto_front.append([obj(x) for obj in objectives])
    
    # Dominance filtering
    pf_array = np.array(pareto_front)
    is_eff = is_pareto(pf_array)
    return pf_array[is_eff]

# Problem 1: Convex Pareto front
true_pf_convex = np.array([[x**2, (x-2)**2] for x in np.linspace(0, 2, 100)])
utopia_convex = np.min(true_pf_convex, axis=0) - 1e-3

# Problem 2: Concave Pareto front 
true_pf_concave = np.array([[x, 1-x**2] for x in np.linspace(-1, 1, 100)])
utopia_concave = np.min(true_pf_concave, axis=0) - [0.1, 1e-3]

# Run algorithm with verification
convex_result = tchebychev_admm(
    objectives=[lambda x: x**2, lambda x: (x-2)**2],
    bounds=(0, 2),
    utopia=utopia_convex
)

concave_result = tchebychev_admm(
    objectives=[lambda x: x, lambda x: 1-x**2],
    bounds=(-1, 1),
    utopia=utopia_concave
)

# Plot results with true Pareto fronts
import matplotlib.pyplot as plt

plt.figure(figsize=(5, 5))
# plt.subplot(121)
plt.scatter(convex_result[:,0], convex_result[:,1], c='r', label='Computed')
plt.plot(true_pf_convex[:,0], true_pf_convex[:,1], 'k--', label='True PF')
plt.title("Convex Pareto Front")
plt.legend()
plt.savefig("convex_pareto_front.png")
plt.close()

plt.figure(figsize=(5, 5))
# plt.subplot(122)
plt.scatter(concave_result[:,0], concave_result[:,1], c='b', label='Computed')
plt.plot(true_pf_concave[:,0], true_pf_concave[:,1], 'k--', label='True PF')
plt.title("Concave Pareto Front")
plt.legend()
plt.savefig("concave_pareto_front.png")
plt.close()

# plt.tight_layout()
# plt.show()