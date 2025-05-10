import numpy as np
import torch
import scipy
from scipy.optimize import LinearConstraint, NonlinearConstraint, Bounds
import matplotlib.pyplot as plt
import sys




# basically what I am gonna do is, I am trying to extend the single-objective admm for multiobjective variant, 
# so I will trying an algorithm to see if it works or not.


# the example I'll be using will be the following:
# JOS1
def f(x):
    n = x.shape
    f1, f2 = (np.linalg.norm(x)**2)/n, (np.linalg.norm(x - 2)**2)/n
    return np.array([f1, f2])

def jac_f(x):
    n = x.shape
    j1 = 2*x/n
    j2 = 2*(x - 2)/n
    return np.vstack([j1, j2])

def g(z):
    return np.zeros(2)

def test_problem():
    x = np.random.rand(3)
    print(f"x: {x}")
    print(f"f(x): {f(x)}")
    print(f"jac_f(x): {jac_f(x)}")
    print(f"g(x): {g(x)}")


def multiobjective_admm(f, jac_f, x0, z0, lambda0,  rho=1, lx=1, lz=1, max_iter=100, tol=1e-6):
    r"""
    We know that multiobjective admm usually looks like:
    .. math::
        \min_{x, z} F(x) + G(z)
        \text{subject to } Ax + Bz = c 
        where F(x) = [f_1(x), f_2(x), ..., f_m(x)] and G(z) = [g_1(z), g_2(z), ..., g_m(z)]
        where m are the number of objectives.

    Since at this point we will be dealing with A=-B=I, c=0, we can simplify the problem to:
    .. math::
        \min_{x, z} F(x) + G(z)
        \text{subject to } x - z = 0
        where F(x) = [f_1(x), f_2(x), ..., f_m(x)] and G(z) = [g_1(z), g_2(z), ..., g_m(z)]
        where m are the number of objectives.

    Now inspired by proximal gradient method for multiobjective optimization, we can write the algorithm as:
    .. math::
        d_x^k = argmin_{d \in \mathbb{R}^n} [max_{i=1,2,...,m} (\nabla f_i(x^k)^Td) + (\lambda^k + \rho(x^k-z^k))^Td + \frac{l}{2}\|d\|^2]\}
        x_{k+1} = x_k + t_x^k *d_x^k
        d_z^k = argmin_{d \in \mathbb{R}^n} [max_{i=1,2,...,m} (g_i(z^k+d) - g_i(z^k)) + (\lambda^k + \rho(x^{k+1}-z^k))^Td + \frac{l}{2}\|d\|^2]\}
        z_{k+1} = z_k + t_x^k *d_z^k
        \lambda_{k+1} = \lambda_k + \rho(x_{k+1} - z_{k+1})



    """

    # Initially I am taking all the g_i's as zero
    # initialize the variables
    # sample problem is JOS1
    xk, zk, lambdak = x0, z0, lambda0
    m, n = f(xk).shape[0], xk.shape[0]
    pr, dr = np.inf, np.inf
    t_x, t_z = 0.01, 0.01 # step sizes, although can use line searches afterwards
    primal_residuals, dual_residuals = [], []
    num_iters = 0

    def objective_x(vars, xk, zk, lambdak):
        # vars = [d, slack]
        return vars[-1] + (lambdak + rho*(xk - zk))@vars[:n] + (lx/2)*np.linalg.norm(vars[:n])**2
    
    def constraints_x(vars, xk, zk, lambdak):
        grad = jac_f(xk)
        return [LinearConstraint(np.hstack([grad, np.array([[-1], [-1]])]), lb=-np.inf, ub=0)]

    for i in range(max_iter):
        print(f"{'-'*50}\nIteration: {i}")
        if pr<tol and dr<tol:
            break
        num_iters += 1

        

        # update x
        vars = np.zeros(n+1)
        d_xk = scipy.optimize.minimize(objective_x, vars, args=(xk, zk, lambdak), constraints=constraints_x(vars, xk, zk, lambdak), 
        method='trust-constr')
        x_k_plus_one = xk + t_x*d_xk.x[:n]

        # update z, g=0
        d_zk = (-1/l)*(lambdak + rho*(x_k_plus_one - zk))
        z_k_plus_one = zk + t_z*d_zk

        # update lambda
        lambdak_plus_one = lambdak + rho*(x_k_plus_one - z_k_plus_one)

        # update residuals
        pr = np.linalg.norm(x_k_plus_one - z_k_plus_one)
        dr = rho*np.linalg.norm(z_k_plus_one - zk)
        primal_residuals.append(pr)
        dual_residuals.append(dr)

        if i%10==0:
            print(f"Iteration: {i}, Primal Residual: {pr}, Dual Residual: {dr}")

        xk, zk, lambdak = x_k_plus_one, z_k_plus_one, lambdak_plus_one

    return xk, zk, lambdak, primal_residuals, dual_residuals, num_iters








    

if __name__ == "__main__":
    np.random.seed(0)
    test_problem()
    print(f"\n\n\n{'-'*50}")

    xk, zk, lambdak, primal_residuals, dual_residuals, num_iters = multiobjective_admm(f, jac_f, np.random.rand(3), np.random.rand(3), np.random.rand(3),
    max_iter=200)
    # we want to show residual plots and pareto front

    print(f"Optimal x: {xk}")
    print(f"Optimal z: {zk}")
    print(f"Optimal lambda: {lambdak}")
    print(f"Number of iterations: {num_iters}")
    plt.subplot(1,2,1)
    plt.plot(list(range(num_iters)), primal_residuals, label='Primal Residuals')
    plt.subplot(1,2,2)
    plt.plot(list(range(num_iters)), dual_residuals, label='Dual Residuals')
    plt.savefig("residuals.png")

    

