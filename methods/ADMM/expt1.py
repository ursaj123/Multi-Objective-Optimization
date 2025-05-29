import numpy as np
from scipy.optimize import minimize

# Define the multi-objective problem (Test Case: Concave Pareto Front)
def f1(x):
    return x[0]

def f2(x):
    return 1 - x[0]**2

def grad_f1(x):
    return np.array([1])

def grad_f2(x):
    return np.array([-2 * x[0]])

# convex pareto front
def f1(x):
    return x[0]**2

def f2(x):
    return (x[0]-1)**2

def grad_f1(x):
    return np.array([2 * x[0]])

def grad_f2(x):
    return np.array([2 * (x[0] - 1)])



# convex pareto front
def f1(x):
    return 0.5*np.linalg.norm(x)**2

def f2(x):
    return 0.5*np.linalg.norm(x-2)**2

def grad_f1(x):
    return x

def grad_f2(x):
    return x - 2

#
def f1(x):
    return np.sin(x[0])

def f2(x):
    return np.sin(x[0]+0.7)

def grad_f1(x):
    return np.array([np.cos(x[0])])

def grad_f2(x):
    return np.array([np.cos(x[0]+0.7)])

# ADMM parameters
def moadmm():
    rho = 1.0
    max_iter = 50
    tol = 1e-6
    # weights = np.array([0.5, 0.5])  # Equal weights for Tchebycheff
    weights = np.random.rand(2)
    weights = weights / np.sum(weights)  # Normalize weights
    z_star = np.array([0.0, 0.0])   # Ideal point (minima of f1 and f2)

    # Initialize variables
    # x = np.array([0.0])  # Initial guess (try different values like 0.0, 0.5, 1.0)
    # x = np.random.rand(1)  # Random initial guess
    x = np.random.uniform(-10, 10, 1)  # Random initial guess
    # print(f1(x), f2(x))
    z = np.array([f1(x), f2(x)])
    lambda_ = np.zeros(2)

    # Storage for history
    history_x = [x.copy()]
    history_z = [z.copy()]
    history_t = []

    for iter in range(max_iter):
        # --- x-update: minimize augmented Lagrangian terms ---
        def objective_x(x_val):
            x = np.array([x_val[0]])
            term1 = (rho/2) * (z[0] - f1(x))**2 - lambda_[0] * f1(x)
            term2 = (rho/2) * (z[1] - f2(x))**2 - lambda_[1] * f2(x)
            return term1 + term2

        def gradient_x(x_val):
            x = np.array([x_val[0]])
            grad1 = rho * (f1(x) - z[0]) * grad_f1(x) - lambda_[0] * grad_f1(x)
            grad2 = rho * (f2(x) - z[1]) * grad_f2(x) - lambda_[1] * grad_f2(x)
            return grad1 + grad2

        # Solve x-update using SciPy's BFGS
        res = minimize(objective_x, x, method='BFGS', jac=gradient_x)
        x_new = res.x
        f_x_new = np.array([f1(x_new), f2(x_new)])
        
        # --- (z, t)-update: closed-form solution ---
        z_unconstrained = f_x_new - lambda_ / rho
        t_candidate = np.max(weights * (z_unconstrained - z_star))
        
        # Project z_i to satisfy w_i(z_i - z_i^*) <= t_candidate
        z_new = np.minimum(z_unconstrained, (t_candidate / weights) + z_star)
        t_new = np.max(weights * (z_new - z_star))  # Ensure t is feasible
        
        # --- Dual update ---
        lambda_new = lambda_ + rho * (z_new - f_x_new)
        
        # Store history
        history_x.append(x_new.copy())
        history_z.append(z_new.copy())
        history_t.append(t_new)
        
        # Check convergence
        primal_residual = np.linalg.norm(z_new - f_x_new)
        dual_residual = rho * np.linalg.norm(z_new - z)
        
        if primal_residual < tol and dual_residual < tol:
            print(f"Converged at iteration {iter}")
            break
        
        # Update variables for next iteration
        x = x_new
        z = z_new
        lambda_ = lambda_new


    print(f"Optimal x: {x}")
    print(f"f1(x): {f1(x):.4f}, f2(x): {f2(x):.4f}")
    return x
# Print final results

f1_approx, f2_approx = [], []
for i in range(50):
    x = moadmm()
    print(x)
    f1_approx.append(f1(x))
    f2_approx.append(f2(x))

# print(f"t: {history_t[-1]:.4f}")

# Plot Pareto front (for visualization)
import matplotlib.pyplot as plt

# Generate Pareto front for the test case
x_pareto = np.linspace(0, 1, 100).reshape(-1, 1)
f1_pareto, f2_pareto = [], []
for i in range(len(x_pareto)):
    f1_pareto.append(f1(x_pareto[i]))
    f2_pareto.append(f2(x_pareto[i]))

plt.figure(figsize=(8, 6))
plt.plot(f1_pareto, f2_pareto, label='Pareto Front', linestyle='--')
plt.scatter(f1_approx, f2_approx)
plt.colorbar(label='Iteration')
plt.xlabel('f1(x)')
plt.ylabel('f2(x)')
plt.title('ADMM Progress on Pareto Front')
plt.legend()
plt.grid(True)
plt.savefig('expt1.png')
plt.show()