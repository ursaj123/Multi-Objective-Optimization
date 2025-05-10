import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import scipy
from scipy.optimize import minimize, NonlinearConstraint, Bounds, LinearConstraint
# from problems import *
from performance_profile import profile
from time import time
style.use('ggplot')



class Newton_and_Grad_Desc_MOP:
    def __init__(self, problem, max_iters=100, sigma = 0.8, tol=1e-6):
        self.max_iters = max_iters
        self.problem = problem
        self.f_list = problem.f_list()
        self.hessians = problem.hessians()
        self.m = problem.m
        self.n = problem.n
        self.sigma = sigma
        self.tol = tol

    def _generate_points(self):
        # Generate points in the feasible region
        x = np.random.randn(self.n)
        # print(f"Initial point x: {x}")
        return x


    def evaluate(self, x):
        # Evaluate the objective functions and constraints at point x
        f_values = []
        for i in range(self.m):
            f_values.append(self.f_list[i][0](x))
        return f_values


    def theta(self, x, s):
        max_term = []
        for i in range(self.m):
            max_term.append(
                np.dot(self.f_list[i][1](x), s) + 0.5*(np.dot(s, np.dot(self.hessians[i](x), s)))
            )
        max_term = np.max(max_term)
        return max_term

    def solve_s_subproblem(self, x):
        values = np.zeros(self.n + 1)
        
        def constraints(values):
            s, t = values[:-1], values[-1]
            temp = np.zeros(self.m)
            for i in range(self.m):
                temp[i] = np.dot(self.f_list[i][1](x), s) + 0.5*(np.dot(s, np.dot(self.hessians[i](x), s))) - t
            return temp
        
        def constraints_jac(values):
            s, t = values[:-1], values[-1]
            jacs = []
            for i in range(self.m):
                jac = np.zeros(self.n + 1)
                jac[:-1] = (self.f_list[i][1](x) + np.dot(self.hessians[i](x), s)).copy()
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
            s, t = values[:-1], values[-1]
            return t

        def objective_jac(values):
            s, t = values[:-1], values[-1]
            jac = np.zeros(self.n + 1)
            jac[:-1] = np.zeros(self.n)
            jac[-1] = 1.0
            return jac

        # Solve the optimization problem
        bounds = self.problem.bounds
        if bounds is not None:
            bounds = tuple([bounds[i] for i in range(self.n)] + [(-np.inf, np.inf)])
        
        res = minimize(
            fun=objective,
            x0=values,
            method='SLSQP',
            jac=objective_jac,
            constraints=non_linear_constraints,
            bounds = bounds,
            # options = {'maxiter':10000}
        )

        if res.success:
            return res.x[:-1]
        else:
            raise ValueError("Optimization failed: " + res.message)

        

    def backtracking(self, x, s, th):
        alpha = 1.0
        num_steps = 0
        f_xk_plus_one = self.evaluate(x + alpha * s)
        f_xk = self.evaluate(x)

        while np.any(f_xk_plus_one > f_xk + self.sigma * alpha * th) and num_steps < 10:
            alpha *= 0.5
            num_steps += 1
            f_xk_plus_one = self.evaluate(x + alpha * s)

            if num_steps == 10:
                print("Backtracking failed to find a suitable step size.")
                break
            
        return alpha

    def solve(self, ):
        start_time = time()
        x = self._generate_points()
        for k in range(self.max_iters):
            print(f"x = {x}")
            s = self.solve_s_subproblem(x)
            th = self.theta(x, s)

            if np.linalg.norm(th) < self.tol:
                break

            # backtracking 
            alpha = self.backtracking(x, s, th)
            x = x + alpha*s

        time_elapsed = time() - start_time
        return x, k, time_elapsed



class JOS1:
    def __init__(self, m=2, n=2, bounds=None, mode='grad_desc'):
        self.m = m
        self.n = n
        self.bounds = None
        self.mode = mode
        # self.bounds = tuple([(-5, 5) for _ in range(n)]) # Assuming 2 variables for JOS1
        # print(self.bounds)

    def f1(self, x):
        return 0.5 * np.linalg.norm(x)**2

    def grad_f1(self, x):
        return x

    def f2(self, x):
        return 0.5 * np.linalg.norm(x - 2)**2

    def grad_f2(self, x):
        return x - 2

    def f_list(self):
        return [
            (self.f1, self.grad_f1),
            (self.f2, self.grad_f2)
        ]

    def hess_f1(self, x):
        return np.eye(self.n)
    
    def hess_f2(self, x):
        return np.eye(self.n)
    
    def hessians(self):
        if self.mode == 'grad_desc':
            return [
                lambda x: np.eye(self.n),
                lambda x: np.eye(self.n)
            ]
        else:
            return [
                self.hess_f1,
                self.hess_f2
            ]

import numpy as np

class Foresnca:
    def __init__(self, m=2, n=2, bounds=None, mode='grad_desc'):
        super().__init__()
        self.m = m
        self.n = n
        self.bounds = None
        self.mode = mode

    def f1(self, x):
        sum1 = sum((xi - 1 / np.sqrt(self.n)) ** 2 for xi in x)
        return 1 - np.exp(-sum1)

    def grad_f1(self, x):
        inv_sqrt_n = 1 / np.sqrt(self.n)
        sum1 = sum((xi - inv_sqrt_n) ** 2 for xi in x)
        return 2 * (np.array(x) - inv_sqrt_n) * np.exp(-sum1)

    def hess_f1(self, x):
        a = 1 / np.sqrt(self.n)
        sum1 = sum((xi - a) ** 2 for xi in x)
        exp_term = np.exp(-sum1)
        n = self.n
        v = np.array(x) - a
        hess = 2 * exp_term * (np.eye(n) - 2 * np.outer(v, v))
        return hess

    def f2(self, x):
        sum2 = sum((xi + 1 / np.sqrt(self.n)) ** 2 for xi in x)
        return 1 - np.exp(-sum2)

    def grad_f2(self, x):
        inv_sqrt_n = 1 / np.sqrt(self.n)
        sum2 = sum((xi + inv_sqrt_n) ** 2 for xi in x)
        return 2 * (np.array(x) + inv_sqrt_n) * np.exp(-sum2)

    def hess_f2(self, x):
        a = 1 / np.sqrt(self.n)
        sum2 = sum((xi + a) ** 2 for xi in x)
        exp_term = np.exp(-sum2)
        n = self.n
        v = np.array(x) + a
        hess = 2 * exp_term * (np.eye(n) - 2 * np.outer(v, v))
        return hess

    def f_list(self):
        return [
            (self.f1, self.grad_f1),
            (self.f2, self.grad_f2)
        ]

    def hessians(self):
        if self.mode == 'grad_desc':
            return [lambda x: np.eye(self.n), lambda x: np.eye(self.n)]
        else:
            return [self.hess_f1, self.hess_f2]



class Circular:
    def __init__(self, m=1, n=2, bounds=None, mode='grad_desc'):
        super().__init__()
        self.m = m
        self.n = n
        self.bounds = None
        self.mode = mode
        cond_num =  1e3
        eigenvalues = np.logspace(0, np.log10(cond_num), n)
    
        # Random orthogonal matrix via QR decomposition of a random matrix
        A = np.random.randn(n ,n)
        Q, _ = np.linalg.qr(A)
        
        # Construct Q = V diag(eigenvalues) V^T (SPD)
        self.Q = Q @ np.diag(eigenvalues) @ Q.T
        self.b = np.random.randn(n)
        self.c = 0.5
        print(f"Original soln: {np.linalg.solve(self.Q, self.b)}")

    def f1(self, x):
        return 0.5 * np.dot(x, self.Q @ x) - np.dot(self.b, x) + self.c
        
    def grad_f1(self, x):
        return self.Q @ x - self.b
        
    def f_list(self):
        return [
            (self.f1, self.grad_f1)
        ]

    def hess_f1(self, x):
        return self.Q   

    def hessians(self):
        if self.mode == 'grad_desc':
            return [lambda x: np.eye(self.n)]
        else:
            return [self.hess_f1]


class Rosenbrock:
    def __init__(self, m=1, n=2, bounds=None, mode='grad_desc'):
        super().__init__()
        self.m = m
        self.n = n
        self.bounds = None
        self.mode = mode

    def f1(self, x):
        return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

    def grad_f1(self, x):
        df1_dx = np.array([-400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]), 200 * (x[1] - x[0]**2)])
        return df1_dx

    def f_list(self):
        return [
            (self.f1, self.grad_f1)
        ]

    def hess_f1(self, x):
        hess = np.zeros((self.n, self.n))
        hess[0, 0] = -400 * x[1] + 1200 * x[0]**2 + 2
        hess[0, 1] = -400 * x[0]
        hess[1, 0] = -400 * x[0]
        hess[1, 1] = 200
        return hess


    def hessians(self):
        if self.mode == 'grad_desc':
            return [lambda x: np.eye(self.n)]
        else:
            return [self.hess_f1]



if __name__=="__main__":
    problems = [
        {
            'problem': JOS1(m=2, n=3, mode='grad_desc'),
            'name': 'JOS1',
            'mode': 'grad_desc',
            'num_sample_points': 200
        },
        {
            'problem': Foresnca(m=2, n=3, mode='grad_desc'),
            'name': 'Foresnca',
            'mode': 'grad_desc',
            'num_sample_points':200
        },
        {
            'problem': Circular(m=1, n=5, mode='grad_desc'),
            'name': 'Circular',
            'mode': 'grad_desc',
            'num_sample_points':1
        },
        {
            'problem': Rosenbrock(m=1, n=2, mode='grad_desc'),
            'name': 'Rosenbrock',
            'mode': 'grad_desc',
            'num_sample_points':1
        }
    ]

    metrics = ['time_elapsed', 'iterations']
    profiles = {'time_elapsed':{'solvers_metrics':[], 'solvers_names':['grad_desc']}, 
    'iterations':{'solvers_metrics':[], 'solvers_names':['grad_desc']}
    }
    # print(profiles)
    # print(profiles['time_elapsed']['solvers_metrics'])
    # sys.exit()

    for item in problems:
        problem = item['problem']
        problem_name = item['name']
        mode = item['mode']
        num_sample_points = item['num_sample_points']
        print(f"Problem: {problem_name}")
        print(f"Mode: {mode}")
        print(f"Number of sample points: {num_sample_points}")
    
        # solver
        number_iterations = []
        f_values = []
        times_elapsed = []
        for i in range(num_sample_points):
            print(f"----------------------Sample point {i + 1}/{num_sample_points}--------------------------")
            solver = Newton_and_Grad_Desc_MOP(problem, max_iters=1000)
            x, k, time_elapsed = solver.solve()
            print(f"Solution: {x}")
            print(f"Iterations: {k}")
            print(f"Time elapsed: {time_elapsed:.4f} seconds")

            # function values
            f_eval = solver.evaluate(x)
            f_values.append(f_eval)
            number_iterations.append(k)
            times_elapsed.append(time_elapsed)
            print('-------------------------------------------------------------------------------------------')



        # performance profile
        profiles['time_elapsed']['solvers_metrics'].append(np.mean(times_elapsed))
        profiles['iterations']['solvers_metrics'].append(np.mean(number_iterations))
        
            
        print(f"Average iterations: {np.mean(number_iterations)}")
        print(f"Average time elapsed: {np.mean(times_elapsed)} seconds")
        # plotting and saving the pareto front only for the functions with two objectives
        if problem.m==2:
            f1_values, f2_values = [], []
            for i in f_values:
                f1_values.append(i[0])
                f2_values.append(i[1])

            plt.scatter(f1_values, f2_values)
            plt.xlabel('f1')
            plt.ylabel('f2')
            plt.title(f'Pareto Front for {problem_name}')

            os.makedirs('newton_results', exist_ok=True)
            plt.savefig(os.path.join('newton_results', f'{problem_name}_{mode}_pareto_front.png'))
            plt.show()


    # profiles
    profile(solvers_metrics = np.array([profiles['time_elapsed']['solvers_metrics']]), 
    solvers_names = profiles['time_elapsed']['solvers_names'], 
    metric_name = 'time elapsed', 
    data_dir = 'newton_results', 
    file_name = 'performance_profile_time.png')

    profile(solvers_metrics = np.array([profiles['iterations']['solvers_metrics']]), 
    solvers_names = profiles['iterations']['solvers_names'], 
    metric_name = 'iterations', 
    data_dir = 'newton_results', 
    file_name = 'performance_profile_iterations.png')
        