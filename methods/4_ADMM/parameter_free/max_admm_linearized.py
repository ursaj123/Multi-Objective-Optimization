import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import scipy
from scipy.optimize import minimize, NonlinearConstraint, Bounds, LinearConstraint
from problems import *
style.use('ggplot')

class SearchDirectionADMM:
    def __init__(self, problem, x, z, rho=1.0, max_iters=100, g_type = 'zero', tol=1e-6, verbose=True):
        # current iterate point
        self.x = x
        self.z = z
        
        # subproblem instance to solve
        self.problem = problem
        self.m = problem.m
        self.n = problem.n
        self.f_list = problem.f_list()

        # ADMM parameters
        self.rho = rho
        self.max_iters = max_iters
        self.verbose = verbose
        self.g_type = g_type
        self.tol = tol

        pass

    def _generate_point(self):
        dx = np.random.rand(self.n)
        dx = dx/np.linalg.norm(dx)
        dz = np.random.rand(self.n)
        dz = dz/np.linalg.norm(dz)
        y = np.random.rand(self.n)
        # print(f"Initial point dx: {dx}")
        return dx, dz, y

    
    def solve_x_subproblem(self, dz, y):
        r"""

        """
        
        values = np.zeros(self.n + 1)
        # warm starting the dx
        values[:-1] = dz.copy()
        values[-1] = max(np.dot(self.f_list[i][1](self.x), dz) for i in range(self.m))

        def linear_constraint():
            A = []
            for i in range(self.m):
                A_i = np.zeros(self.n + 1)
                A_i[:-1] = self.f_list[i][1](self.x)
                A_i[-1] = -1
                A.append(A_i)

            return np.array(A)

        def non_linear_constraint(values):
            dx, s = values[:-1], values[-1]
            return np.linalg.norm(dx) - 1

        def non_linear_jac(values):
            dx, s = values[:-1], values[-1]
            jac = np.zeros(self.n + 1)
            jac[:-1] = dx/np.linalg.norm(dx)
            jac[-1] = 0
            return jac 

        def objective_function(values):
            dx, s = values[:-1], values[-1]
            return s + np.dot(y, self.x + dx - self.z - dz) + (self.rho/2)*np.linalg.norm(
                self.x + dx - self.z - dz)**2

        def objective_jac(values):
            dx, s = values[:-1], values[-1]
            jac = np.zeros(self.n + 1)
            jac[:-1] = (y + self.rho*(self.x + dx - self.z - dz)).copy()
            jac[-1] = 1
            return jac

        linear_constraint = LinearConstraint(A=linear_constraint(), lb=-np.inf, ub=0.0)
        non_linear_constraint = NonlinearConstraint(
            fun=non_linear_constraint,
            jac=non_linear_jac,
            lb=-np.inf,
            ub=0.0
        )
        # cons = [
        #     # {'type':'ineq', 'fun':non_linear_constraint, 'jac':non_linear_jac},
        #     {'type':'ineq', 'fun':linear_constraint}
        # ]
        cons = [
            linear_constraint,
            non_linear_constraint
        ]
        # # gotta check the bounds also, they can be like -5-x<dx<5-x

        res = minimize(
            fun=objective_function,
            x0=values,
            method='trust-constr',
            jac=objective_jac,
            constraints=cons,
            # options={'disp': False, 'maxiter': self.max_iters}
        )

        if res.success:
            return res.x[:-1]
        else:
            raise ValueError("Optimization failed: " + res.message)

    def solve_z_subproblem(self, dx, y):
        r"""
        Currently only handling the case where g is zero, L1, or indicator function
        """
        if g_type=='zero':
            return self.x + dx - self.z + y/self.rho
        elif g_type=='L1':
            pass
        elif g_type=='indicator':
            pass
    
    def solve(self):
        dxk, dzk, yk = self._generate_point()
        history = {'primal':[], 'dual':[]}
        for i in range(self.max_iters):
            print(f"----------------------SearchDirection ADMM Iteration {i}---------------------------------")
            # solve the x subproblem
            dxk_plus_one = self.solve_x_subproblem(dzk, yk)

            # solve the z subproblem
            dzk_plus_one = self.solve_z_subproblem(dxk_plus_one, yk)

            # update the multipliers
            yk_plus_one = yk + self.rho * (self.x + dxk_plus_one - self.z - dzk_plus_one)

            # check for convergence
            primal_residual = np.linalg.norm(self.x + dxk_plus_one - self.z - dzk_plus_one)
            dual_residual = self.rho*np.linalg.norm(dzk_plus_one - dzk)

            if primal_residual < self.tol and dual_residual < self.tol:
                if self.verbose:
                    print(f"Sub-Problem ADMM converged at iteration {i}, primal residual = {primal_residual}, dual residual = {dual_residual}")
                break

            print(f"Search Direction ADMM iteration {i}\ndxk = {dxk}\ndzk = {dzk}:\nprimal_residual = {primal_residual}, dual_residual = {dual_residual}")
            # update the variables
            dxk = dxk_plus_one
            dzk = dzk_plus_one
            yk = yk_plus_one

            # update the history
            history['primal'].append(primal_residual)
            history['dual'].append(dual_residual)

        return dxk_plus_one, dzk_plus_one, yk_plus_one, history



    
class LinearizedAdmmMOP:
    def __init__(self, problem, g_type = 'zero', step_size = 1e-2, max_iter_linearized_admm = 20, 
                        max_iter_search_dir_admm=100, rho=1.0, tol=1e-6, verbose=True):
        self.problem = problem
        self.m = problem.m
        self.n = problem.n
        self.f_list = problem.f_list()
        self.g_type = g_type


        self.max_iter_linearized_admm = max_iter_linearized_admm
        self.max_iter_search_dir_admm = max_iter_search_dir_admm
        self.step_size = step_size
        self.rho = rho
        self.tol = tol
        self.verbose = verbose
        pass

    def _generate_point(self):
        x = np.random.rand(self.n)
        z = np.random.rand(self.n)
        return x, z


    def evaluate(self, x, z):
        r"""
        Evaluate the function values at the point x and z
        """
        f_values = []
        for i in range(self.m):
            f_values.append(self.f_list[i][0](x))

        if self.g_type == 'zero':
            pass
        elif self.g_type == 'L1':
            pass
        elif self.g_type == 'indicator':
            pass
        
        return f_values


    def armijo_backtrack(self, x, z, dx, dz, sigma = 0.8):
        alpha = 1.0
        # Compute the function value at the current point
        f_values_k_plus_one = self.evaluate(x+alpha*dx, z+alpha*dz)
        f_values_k = self.evaluate(x, z)

        # 
        max_term = []
        for i in range(self.m):
            term = np.dot(self.f_list[i][1](x), dx)
            if self.g_type == 'zero':
                pass
            elif self.g_type == 'L1':
                pass
            elif self.g_type == 'indicator':
                pass
            max_term.append(term)

        max_term = max(max_term)

        # Compute the Armijo condition
        number_of_evals = 0
        while np.any(f_values_k_plus_one > f_values_k + alpha*sigma*max_term) and number_of_evals < 10:
            alpha *= 0.5
            f_values_k_plus_one = self.evaluate(x+alpha*dx, z+alpha*dz)
            number_of_evals += 1

            # Check if the Armijo condition was satisfied
            if number_of_evals == 10:
                if self.verbose:
                    print("Armijo backtracking failed to converge")
                break

        if self.verbose:
            print(f"Armijo backtracking converged at alpha = {alpha}, number of evaluations = {number_of_evals}")

        return alpha
    
    def solve(self):
        x, z = self._generate_point()
        norms = []
        for i in range(self.max_iter_linearized_admm):
            # solve the search direction problem
            print(f"----------------------Linearized ADMM Iteration {i}---------------------------------")
            print(f"x = {x}, z = {z}\n")
            search_direction_admm = SearchDirectionADMM(
                problem=self.problem,
                x=x,
                z=z,
                rho=self.rho,
                max_iters=self.max_iter_search_dir_admm,
                g_type=self.g_type,
                tol=self.tol,
                verbose=self.verbose
            )
            dx, dz, y, history = search_direction_admm.solve()

            # update the variables, must implement the projection and bounds afterwards
            alpha = self.armijo_backtrack(x, z, dx, dz)
            x = x + alpha*dx
            z = z + alpha*dz

            # check for convergence
            norm = np.sqrt(np.linalg.norm(dx)**2 + np.linalg.norm(dz)**2)
            if norm < self.tol:
                if self.verbose:
                    print(f"Norm Converged at iteration {i}, norm = {norm}")
                break

            # update the norms
            norms.append(norm)
            # print(f"----------------------------------------------------------------------")

        return x, z, norms
        


if __name__=="__main__":
    # problem = Poloni()
    # problem_name = 'Poloni'
    problem = Foresnca()
    problem_name = 'Foresnca'
    # problem = ZDT1()
    # problem_name = 'ZDT1'
    # problem = Rosenbrock()
    # problem_name = 'Rosenbrock'
    # problem = Easom()
    # problem_name = 'Easom'
    # problem = Circular()
    # problem_name = 'circular'
    # problem = JOS1()
    # problem_name = 'JOS1'
    num_sample_points = 1
    g_type = 'zero'
    max_iter_search_dir_admm = 100
    max_iter_linearized_admm = 10
    step_size = 1e-0
    rho = 1
    tol = 1e-4
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
        print(f"-------------------Sampling point number {i} -------------------------")
        linearized_admm = LinearizedAdmmMOP(
            problem=problem,
            g_type=g_type,
            step_size=step_size,
            max_iter_linearized_admm=max_iter_linearized_admm,
            max_iter_search_dir_admm=max_iter_search_dir_admm,
            rho=rho,
            tol=tol,
            verbose=verbose
        )
        x, z, norms = linearized_admm.solve()
        print(f"Converged at x = {x}")
        print(f"Converged at z = {z}")
        print(f"Norms progression = {norms}")
        # print(f"x, z: {x}, {z}")

        # function values
        # print(problem.f_list()[0][1](x))
        func_eval = linearized_admm.evaluate(x, z)
        # func_eval = admm.evaluate(z, x)
        func_evals.append(func_eval)
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
        plt.savefig(f'{problem_name} Pareto Front,  g = {g_type}_linearized_max.png')


    pass


# subproblem is too much dependent on x and z


