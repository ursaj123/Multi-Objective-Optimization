import numpy as np
from scipy.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

def generate_weights(n_weights):
    """Generate weights using simplex-lattice design"""
    return np.array([[i/(n_weights-1), 1-i/(n_weights-1)] for i in range(n_weights)])

def tchebychev_admm(objectives, bounds, utopia, n_weights=50, rho=1.0, max_iter=200):
    """Tchebychev-ADMM with proper Pareto front generation"""
    weights = generate_weights(n_weights)
    pareto_front = []
    
    for λ in weights:
        # Initialize variables for each weight
        x = np.random.uniform(bounds[0], bounds[1])
        z = x
        u = 0.0
        
        for _ in range(max_iter):
            # X-update: Solve Tchebychev scalarization
            def x_obj(var):
                x_val = var[0]
                alpha = var[1]
                f = [obj(x_val) for obj in objectives]
                constraints = [
                    λ[0] * (f[0] - utopia[0]) - alpha,
                    λ[1] * (f[1] - utopia[1]) - alpha
                ]
                penalty = 1e4 * sum(np.maximum(0, constraints)**2)
                return alpha + 0.5*rho*(x_val - z + u)**2 + penalty
            
            res = minimize(
                x_obj,
                [x, 0],
                bounds=[(bounds[0], bounds[1]), (None, None)],
                method="SLSQP",
                options={"ftol": 1e-12}
            )
            x_new = res.x[0]
            
            # Z-update with projection
            z_new = np.clip(x_new + u, bounds[0], bounds[1])
            
            # Dual update
            u += x_new - z_new
            
            if abs(x_new - z_new) < 1e-9:
                break
            
            x, z = x_new, z_new
        
        pareto_front.append([obj(x) for obj in objectives])
    
    # Dominance filtering
    pf_array = np.array(pareto_front)
    return pf_array
    front = NonDominatedSorting().do(pf_array, only_non_dominated=True)
    # print(f"Front: {front}")
    # return pf_array
    # return pf_array[front[0]]
    return pf_array[front]

# ------------------------------------------------------------
# Test Cases
# ------------------------------------------------------------
if __name__ == "__main__":
    # Convex Pareto front
    convex_pf = tchebychev_admm(
        objectives=[lambda x: x**2, lambda x: (x-2)**2],
        bounds=(-2, 2),
        utopia=[-1e-3, -1e-3]
    )
    
    # Concave Pareto front (note f2 = -(1-x^2) for minimization)
    concave_pf = tchebychev_admm(
        objectives=[lambda x: x, lambda x: -(1 - x**2)],
        bounds=(0, 1),
        utopia=[-0.1, -1.1]
    )
    
    # Plot results
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(5, 5))
    plt.scatter(convex_pf[:,0], convex_pf[:,1], c="red", label="Computed")
    plt.plot([x**2 for x in np.linspace(0, 2, 100)], 
             [(x-2)**2 for x in np.linspace(0, 2, 100)], "k--", label="True PF")
    plt.xlabel("$f_1 = x^2$"), plt.ylabel("$f_2 = (x-2)^2$")
    plt.legend()
    plt.savefig("convex_pareto_front.png")

    plt.figure(figsize=(5, 5))
    plt.scatter(concave_pf[:,0], -concave_pf[:,1], c="blue", label="Computed")  # Revert sign for f2
    plt.plot([x for x in np.linspace(-1, 1, 100)], 
             [1 - x**2 for x in np.linspace(-1, 1, 100)], "k--", label="True PF")
    plt.xlabel("$f_1 = x$"), plt.ylabel("$f_2 = 1-x^2$")
    plt.legend()
    plt.savefig("concave_pareto_front.png")
    
    # plt.tight_layout()
    # plt.show()