import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize, minimize_scalar, LinearConstraint, Bounds
from time import time
from jaxopt.projection import projection_box
from jaxopt.prox import prox_lasso
from matplotlib import pyplot as plt
# np.random.seed(0)

def prox_wsum_g():
    """
        the dual problem to solve is:
            
        is only for L1 norm
        where g_i(x) = ||x-i+1||_1/(i*n)

        We will get D_z after solving this, where d_z = D_z - z
        and remember that d_z can be bounded by Linf norms
        if self.norm_constraint == 'Linf':
            then ||d_z||_inf<=1 implies ||D_z||_inf = ||d_z + z||_inf <= 1 + ||z||_inf, thus we can put such bounds on D_z
            and then, after finding out solutions for D_z, we will find out the solution for d_z
        else:
            there is no constraint on D_z



        basic test cases
        problem:
            argmin_z(||z||_1 + ||x-z||_2^2)

        for, this should be same as Shrinking operator
        as np.sign(x) * np.maximum(np.abs(x) - 1, 0)
        (verified it, working well)

        problem:
            argmin_z(||z||_1/2 + ||z-1||_1/4 + ||x-z||_2^2)
        taking weights, l1_ratios, l1_shifts = [1,1], [0.5, 0.25], [0, 1]
        initial point x = [-1. , -0.5,  0.3,  1. ,  1.5,  2. ]
        the answer should be [-0.25,  0.  ,  0.05,  0.75,  1.  ,  1.25]

        closed form solution to it can be found via this function
        prox_f = lambda x: np.where(x < -0.75, x + 0.75,
                    np.where(x < 0.25, 0,
                    np.where(x < 1.25, x - 0.25,
                    np.where(x < 1.75, 1, x - 0.75))))
        (verified, working well)
        """
    def prox_lasso_(weights, x, l1_ratios, l1_shifts, m, bounds=None):
        coef = weights * l1_ratios
        x = prox_lasso(
            x + np.sum(coef[1:]) - l1_shifts[0] + l1_shifts[0], coef[0]
        )
        for i in range(1, m):
            x = (
                prox_lasso(x - coef[i] - l1_shifts[i], coef[i])
                + l1_shifts[i]
            )

        if bounds is not None:
            x = projection_box(x, (bounds[0], bounds[1]))
        return x
    
    # let us take a random x
    x = np.random.randn(5)
    print("x:", x)
    # in case m=1, the prox operator of x is shrinkage operator

    print('Testing for m=1 case')
    weights, l1_ratios, l1_shifts, bounds = np.array([1.0]), np.array([1.0]), np.array([0.0]), None
    prox_shrinkage = lambda x: np.sign(x) * np.maximum(np.abs(x) - 1, 0)
    res_prox_shrinkage = prox_shrinkage(x)
    res_prox_lasso = prox_lasso_(weights=weights, x=x, l1_ratios=l1_ratios, l1_shifts=l1_shifts, m=1, bounds=bounds)

    print("prox_shrinkage(x):", res_prox_shrinkage)
    print("prox_lasso(x):", res_prox_lasso)

    print('Testing for m=2 case')
    l1_ratios, l1_shifts, weights = np.array([0.5, 0.25]), np.array([0, 1]), np.array([1, 1])

    prox_f = lambda x: np.where(x < -0.75, x + 0.75,
                    np.where(x < 0.25, 0,
                    np.where(x < 1.25, x - 0.25,
                    np.where(x < 1.75, 1, x - 0.75))))

    res_prox_f = prox_f(x)
    res_prox_lasso = prox_lasso_(weights=weights, x=x, l1_ratios=l1_ratios, l1_shifts=l1_shifts, m=2, bounds=bounds)
    print("prox_f(x):", res_prox_f)
    print("prox_lasso(x):", res_prox_lasso)
    pass



def minimize_scalar_test():
    """
    problem:
        max_{l} argmin_z(l*||z||_1/2 + (1-l)*||z-1||_1/4 + ||x-z||_2^2), s.t. 0<=l<=1
    """

    def prox_piecewise(xi, lam):
        a = lam / 2
        b = (1 - lam) / 4
        # a,b = 0.5, 0.25
        T1, T2 = -(a + b), a - b
        T3, T4 = 1 + a - b, 1 + a + b

        if xi < T1:
            return xi + (a + b)
        elif xi < T2:
            return 0
        elif xi < T3:
            return xi - (a - b)
        elif xi < T4:
            return 1
        else:
            return xi - (a + b)

    vectorized_prox = np.vectorize(prox_piecewise)

    x = np.random.randn(5)*-1
    def func_val(lam):
        z = vectorized_prox(x, lam)
        return - (lam / 2 * norm(z, ord=1) + (1 - lam) / 4 * norm(z - 1, ord=1) + norm(x - z, ord=2) ** 2)
    # print("x:", x)
    # print("prox_piecewise(x):", vectorized_prox(x, 0.5))


    weights = np.linspace(0, 1, 200)
    values = []
    for i in weights:
        values.append(-func_val(i))
    values = np.array(values)
    argmin = np.argmin(values)

    print(f"Optimal lambda grid search: {weights[argmin]}")

    plt.plot(weights, values)
    plt.savefig('minimize_scalar_test.png')

    res = minimize_scalar(func_val, bounds=(0, 1))
    print(f"Optimal lambda minimize scalar: {res.x}")


    pass



# print('Testing prox_wsum_g')
# prox_wsum_g()

# print('Testing minimize_scalar')
# minimize_scalar_test()

