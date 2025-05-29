import numpy as np
from scipy.optimize import minimize

# --------------------------------
# Tchebychev-ADMM Core Algorithm
# --------------------------------

def tchebychev_admm(objectives, bounds, utopia, n_weights=10, rho=1.0, max_iter=100):
    """
    Tchebychev-ADMM for multi-objective optimization
    """
    n_objs = len(objectives)
    weights = np.array([[i/(n_weights-1), 1-i/(n_weights-1)] for i in range(n_weights)])
    
    pareto_front = []
    
    for λ in weights:
        # ADMM variables
        x = np.mean(bounds, axis=1)  # Initialization within bounds
        z = x.copy()
        u = np.zeros_like(x)
        
        for _ in range(max_iter):
            # X-update: Solve Tchebychev scalarized problem
            def x_obj(x_var):
                f = np.array([obj(x_var) for obj in objectives])
                tcheby = np.max(λ * (f - utopia))
                return tcheby + (rho/2)*np.linalg.norm(x_var - z + u)**2
            
            res = minimize(x_obj, x, bounds=bounds)
            x_new = res.x
            
            # Z-update with projection to feasible region
            z_new = np.clip(x_new + u, *zip(*bounds))
            
            # Dual update
            u += x_new - z_new
            
            if np.linalg.norm(x_new - z_new) < 1e-6:
                break
                
            x, z = x_new, z_new
            
        pareto_front.append([obj(x) for obj in objectives])
    
    return np.array(pareto_front)

# --------------------------------
# Test Problems
# --------------------------------

# Problem 1: Convex Pareto front
def problem1_convex(x):
    return [x[0]**2, (x[0]-2)**2]

# Problem 2: Concave Pareto front
def problem2_concave(x):
    return [x[0], 1 - x[0]**2]

# --------------------------------
# Run Algorithm
# --------------------------------
if __name__ == "__main__":
    # Problem 1: Convex front (x ∈ [0, 2])
    convex_front = tchebychev_admm(
        objectives=[lambda x: problem1_convex(x)[0], 
                    lambda x: problem1_convex(x)[1]],
        bounds=[(0, 2)],
        utopia=[0, 0],  # (x=0, x=2)
        n_weights=20
    )
    print(f"Convex Pareto front: {convex_front}")
    
    # Problem 2: Concave front (x ∈ [-1, 1])
    concave_front = tchebychev_admm(
        objectives=[lambda x: problem2_concave(x)[0], 
                    lambda x: problem2_concave(x)[1]],
        bounds=[(-1, 1)],
        utopia=[-1, 0],  # (x=-1 f1, x=0 f2)
        n_weights=20
    )
    print(f"Concave Pareto front: {concave_front}")
    
    # Plot results
    import matplotlib.pyplot as plt
    from matplotlib import style
    style.use('ggplot')

    plt.figure(figsize=(5, 5))
    
    # Convex front
    # plt.subplot(121)
    plt.scatter(convex_front[:,0], convex_front[:,1], c='r')
    plt.title("Convex Pareto Front")
    plt.xlabel("f1 = x²")
    plt.ylabel("f2 = (x-2)²")
    plt.savefig("convex_front.png")
    plt.close()
    
    # Concave front
    plt.figure(figsize=(5, 5))

    # plt.subplot(122)
    plt.scatter(concave_front[:,0], concave_front[:,1], c='b')
    plt.title("Concave Pareto Front")
    plt.xlabel("f1 = x")
    plt.ylabel("f2 = 1-x²")
    plt.savefig("concave_front.png")
    plt.close()
    
    # plt.tight_layout()
    # plt.show()