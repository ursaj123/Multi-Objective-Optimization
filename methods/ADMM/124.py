import numpy as np
from scipy.optimize import minimize

def tchebychev_admm(objectives, bounds, utopia, n_weights=10, rho=1.0, eps=1e-3, max_iter=100):
    """
    Tchebychev-ADMM for multi-objective optimization.
    
    :param objectives: List of objective functions [f1, f2]
    :param bounds: Variable bounds [(lower, upper)]
    :param utopia: Utopia point [y1*, y2*]
    :param n_weights: Number of weight vectors
    :param rho: ADMM penalty parameter
    :param eps: Utopia point offset
    :param max_iter: Maximum iterations
    :return: Pareto front points
    """
    n_objs = len(objectives)
    weights = np.array([[w, 1-w] for w in np.linspace(0, 1, n_weights)])
    pareto_front = []
    
    for λ in weights:
        # ADMM variables
        x = np.random.uniform(bounds[0][0], bounds[0][1])
        z = x
        u = 0.0
        
        for _ in range(max_iter):
            # X-update: Solve Tchebychev scalarization with slack variable
            def x_update(var):
                x_val = var[0]
                alpha = var[1]
                f = [obj(x_val) for obj in objectives]
                constraints = [
                    λ[0] * (f[0] - utopia[0]) - alpha,
                    λ[1] * (f[1] - utopia[1]) - alpha
                ]
                penalty = 0.5 * rho * (x_val - z + u)**2
                return alpha + penalty + 1e3 * sum(np.maximum(0, constraints)**2)  # Penalty for constraints
            
            res = minimize(x_update, [x, 0], bounds=bounds + [(None, None)])
            x_new, alpha_new = res.x[0], res.x[1]
            
            # Z-update
            if bounds[0][0] is not None or bounds[0][1] is not None:
                z_new = np.clip(x_new + u, bounds[0][0], bounds[0][1])
            else:
                z_new = x_new + u
            
            # Dual update
            u += x_new - z_new
            
            if np.abs(x_new - z_new) < 1e-6:
                break
            
            x, z = x_new, z_new
        
        pareto_front.append([obj(x) for obj in objectives])
    
    return np.array(pareto_front)

# --------------------------------
# Test Cases
# --------------------------------
if __name__ == "__main__":
    # Problem 1: Convex Pareto front (x ∈ [0, 2])
    convex_pf = tchebychev_admm(
        objectives=[lambda x: x**2, lambda x: (x-2)**2],
        bounds=[(-2, 2)],
        utopia=[-0.1, -0.1],  # Slightly below true minima
        n_weights=100
    )
    
    # Problem 2: Concave Pareto front (x ∈ [-1, 1])
    concave_pf = tchebychev_admm(
        objectives=[lambda x: x, lambda x: 1 - x**2],
        bounds=[(0, 1)],
        utopia=[-0.1, -0.1],  # Slightly below minima
        n_weights=20
    )
    
    # Plot results
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(5, 5))
    plt.scatter(convex_pf[:,0], convex_pf[:,1], c='red')
    plt.title("Convex Pareto Front: $f_1 = x^2,\ f_2 = (x-2)^2$")
    plt.xlabel("$f_1$"), plt.ylabel("$f_2$")
    plt.savefig("convex_pareto_front.png")
    plt.close()
    
    plt.figure(figsize=(5, 5))
    plt.scatter(concave_pf[:,0], concave_pf[:,1], c='blue')
    plt.title("Concave Pareto Front: $f_1 = x,\ f_2 = 1-x^2$")
    plt.xlabel("$f_1$"), plt.ylabel("$f_2$")
    plt.savefig("concave_pareto_front.png")
    plt.close()
    
    # plt.tight_layout()
    # plt.show()