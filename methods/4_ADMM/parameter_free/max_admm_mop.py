
import numpy as numpy
import sys
import os
import scipy
from scipy.optimize import minimize, NonlinearConstraint
from problems import *
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

class ADMM_MOP:
    r"""
    Initially we are only dealing with the cases 
    of x=z where A=-B=I, c=0
    """
    def __init__(self, problem, g_type='zero', max_iter=100, rho=1.0, tol=1e-6, verbose=True):
        r"""
        
        Parameters
        ----------
        m : int
            Number of objectives.
        n : int
            Number of variables.
        problem : Problem
            Problem instance.
        g_type : str
            Type of non-differentiable part, it can be one of 0, L1, or indicator function
        max_iter : int
            Maximum number of iterations.
        rho : float
            Penalty parameter.
        tol : float
            Tolerance for convergence.
        verbose : bool
            If True, print convergence messages.
        """

        self.max_iter = max_iter
        self.rho = rho
        self.tol = tol
        self.problem = problem
        self.m = problem.m
        self.n = problem.n
        self.g_type = g_type
        self.verbose = verbose
        self.f_list = problem.f_list()
        pass

    def _generate_point(self):
        x = np.random.rand(self.n)
        z = np.random.rand(self.n)
        y = np.random.rand(self.n)
        print(f"Initial point x: {x}")
        return x, z, y

    def solve_x_subproblem(self, z, y):
        r"""
        Solve the x subproblem for the ADMM algorithm.
        The subproblem is defined as:
        .. math::
            x^{k+1} = argmin_x [(max_{i = 1,2,...,m} f_i(x)) + \lambda^T(Ax+Bz-c) + \frac{\rho}{2}\|Ax + Bz-c\|_2^2]

        to deal with the max function, we can solve the constrained optimization problem given by:
        .. math::
            x^{k+1} = argmin_x [s + \lambda^T(Ax+Bz-c) + \frac{\rho}{2}\|Ax + Bz-c\|_2^2] \\ s.t. f_i(x) \leq s, i = 1,2,...,m

        where s is a slack variable.
        """
        # values = np.random.randn(self.n + 1)
        values = np.zeros(self.n + 1)
        # values[:-1] = z.copy()
        # values[-1] = max(self.f_list[i][0](z) for i in range(self.m))
        # print(values)

        def constraints(values):
            temp = np.zeros(self.m)
            for i in range(self.m):
                temp[i] = self.f_list[i][0](values[:-1]) - values[-1]
            
            return temp

        def constraints_jac(values):
            jacs = []
            for i in range(self.m):
                jac = np.zeros(self.n + 1)
                jac[:-1] = (self.f_list[i][1](values[:-1])).copy()
                jac[-1] = -1
                jacs.append(jac)
            
            return np.array(jacs)

        non_linear_constraints = NonlinearConstraint(
            fun = constraints,
            jac = constraints_jac,
            lb = -np.inf,
            ub = 0.0
        )


        def objective(values):
            return values[-1] + np.dot(y, values[:-1]-z) + (self.rho/2)*np.linalg.norm(values[:-1] - z)**2

        def objective_jac(values):
            jac = np.zeros(self.n + 1)
            jac[:-1] = (y + self.rho*(values[:-1] - z)).copy()
            jac[-1] = 1.0
            return jac
        
        # Solve the optimization problem
        bounds = self.problem.bounds
        if bounds is not None:
            bounds = tuple([bounds[i] for i in range(self.n)] + [(-np.inf, np.inf)])
        res = minimize(
            fun=objective,
            x0=values,
            method='trust-constr',
            # method = 'SLSQP',
            jac=objective_jac,
            constraints=non_linear_constraints,
            bounds = bounds,
            # options = {'maxiter':10000}
        )

        if res.success:
            return res.x[:-1]
        else:
            raise ValueError("Optimization failed: " + res.message)




    def solve_z_subproblem(self, x, y):
        r"""
        """
        if self.g_type == 'zero':
            return x + y/self.rho
        elif self.g_type == 'l1':
            pass
        elif self.g_type == 'indicator':
            pass


    def evaluate(self, x, z):
        r"""
        Compute the function values for the given point x.
        """
        f_values = np.zeros(self.m)
        for i in range(self.m):
            f_values[i] = self.f_list[i][0](x)
        
        # Add the non-differentiable part
        if self.g_type == 'zero':
            pass
        elif self.g_type == 'l1':
            pass
        elif self.g_type == 'indicator':
            pass
        
        return f_values

    def solve(self):
        xk, zk, yk = self._generate_point()
        history = {'primal':[], 'dual':[]}

        for i in range(self.max_iter):
            # print(f"i = {i}, xk = {xk}")
            xk_plus_one = self.solve_x_subproblem(zk, yk)
            zk_plus_one = self.solve_z_subproblem(xk_plus_one, yk)
            yk_plus_one = yk + self.rho * (xk_plus_one - zk_plus_one)

            # Check convergence
            primal_residual = np.linalg.norm(xk_plus_one - zk_plus_one)
            # dual_residual = self.rho*np.sqrt(self.m)*np.linalg.norm(zk_plus_one - zk)
            dual_residual = self.rho*np.linalg.norm(zk_plus_one - zk)
            history['primal'].append(primal_residual)
            history['dual'].append(dual_residual)
            if primal_residual < self.tol and dual_residual < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {i}")
                break

            # Update variables
            xk = xk_plus_one
            zk = zk_plus_one
            yk = yk_plus_one

        return xk_plus_one, zk_plus_one, yk_plus_one, history
            


        pass


if __name__=="__main__":
    # problem = Poloni()
    # problem_name = 'Poloni'
    # problem = Foresnca()
    # problem_name = 'Foresnca'
    # problem = ZDT1()
    # problem_name = 'ZDT1'
    # problem = Rosenbrock()
    # problem_name = 'Rosenbrock'
    # problem = Easom()
    # problem_name = 'Easom'
    # problem = Circular()
    # problem_name = 'circular'
    problem = JOS1()
    problem_name = 'JOS1'
    num_sample_points = 10
    g_type = 'zero'
    max_iter = 100
    rho = 0.01
    tol = 1e-6
    verbose = True

    # admm = ADMM_MOP(
    #         problem=problem,
    #         g_type=g_type,
    #         max_iter=max_iter,
    #         rho=rho,
    #         tol=tol,
    #         verbose=verbose
    #     )
    
    # x = np.array([0.99999415, 1.00000582])
    # print(admm.evaluate(x, x))

    # x = np.array([0, 2.0])
    # print(admm.evaluate(x, x))

    # # x = np.array([])

    # sys.exit()
    func_evals = []

    for i in range(num_sample_points):
        print(f"-------------------Iteration {i} -------------------------")
        admm = ADMM_MOP(
            problem=problem,
            g_type=g_type,
            max_iter=max_iter,
            rho=rho,
            tol=tol,
            verbose=verbose
        )
        x, z, y, history = admm.solve()
        print(f"Converged at x = {x}")
        print(f"Converged at z = {z}")
        # print(f"x, z: {x}, {z}")

        # function values
        # print(problem.f_list()[0][1](x))
        # func_eval = admm.evaluate(x, z)
        func_eval = admm.evaluate(z, x)
        func_evals.append(func_eval)
        print(f"history = {history}")

        print('------------------------------------------------------------')

        
    # print(func_evals)
    # plotting the pareto front only for the functions with two objectives
    if problem.m==2:
        f1_values, f2_values = [], []
        for i in func_evals:
            f1_values.append(i[0])
            f2_values.append(i[1])

        print(f1_values)
        print(f2_values)
        plt.figure(figsize=(5, 5))
        plt.scatter(f1_values, f2_values, color='blue') # , label='L1 Regularization'
        plt.title(f'{problem_name} Pareto Front,  g = {g_type}')
        plt.xlabel('f1(x)')
        plt.ylabel('f2(x)')
        plt.grid('True')
        plt.show()

        # Save the plot
        plt.savefig(f'{problem_name} Pareto Front,  g = {g_type}_max.png')
        pass


    pass
