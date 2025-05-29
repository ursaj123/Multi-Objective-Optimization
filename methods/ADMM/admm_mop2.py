import numpy as np
from scipy.optimize import minimize

class MO_ADMM:
    def __init__(self, problem, rho=1.0, alpha=0.1, beta=0.9, tol=1e-6, max_iter=100):
        """
        Multi-Objective ADMM Solver
        """
        self.problem = problem
        self.rho = rho
        self.alpha = alpha  # Regularization parameter
        self.beta = beta    # Regularization decay
        self.tol = tol
        self.max_iter = max_iter
        self.history = []
        
        # Initialize variables
        self.x = np.random.uniform(problem.lb, problem.ub, problem.n)
        self.z = np.random.uniform(problem.lb, problem.ub, problem.n)
        # self.u = [np.zeros(problem.n) for _ in range(problem.m)]
        self.u = [np.random.uniform(problem.lb, problem.ub, problem.n) for _ in range(problem.m)]

    def _x_update(self):
        """Pareto-optimal x-update step"""
        def obj_func(vars):
            x, t = vars[:self.problem.n], vars[self.problem.n]
            return t
            
        constraints = []
        for i in range(self.problem.m):
            f, grad_f, _ = self.problem.f_list()[i]
            constraints.append({
                'type': 'ineq',
                'fun': lambda vars, i=i: (
                    -f(vars[:self.problem.n]) 
                    - self.rho/2 * np.linalg.norm(vars[:self.problem.n] - self.z + self.u[i])**2
                    - self.alpha/2 * np.linalg.norm(vars[:self.problem.n] - self.x)**2
                    + vars[self.problem.n]
                )
            })
            
        res = minimize(obj_func, 
                       x0=np.concatenate([self.x, [0]]), 
                       constraints=constraints,
                       bounds=self.problem.bounds + [(None, None)])
        return res.x[:self.problem.n]

    def _z_update(self):
        """Consensus z-update step with g-function handling"""
        if self.problem.g_type[0] == 'zero':
            # Closed-form solution for quadratic case
            return np.mean([self.x + ui for ui in self.u], axis=0)
            
        elif self.problem.g_type[0] == 'L1':
            # Proximal operator for L1 regularization
            gamma = self.problem.g_type[1].get('lambda', 0.1)
            temp = np.mean([self.x + ui for ui in self.u], axis=0)
            return np.sign(temp) * np.maximum(np.abs(temp) - gamma/self.rho, 0)
            
        elif self.problem.g_type[0] == 'indicator':
            # Projection onto feasible set
            temp = np.mean([self.x + ui for ui in self.u], axis=0)
            return np.clip(temp, self.problem.lb, self.problem.ub)
            
        else:
            raise ValueError(f"Unsupported g_type: {self.problem.g_type[0]}")

    def _update_dual_variables(self, x_new, z_new):
        """Dual variable update for each objective"""
        for i in range(self.problem.m):
            self.u[i] += x_new - z_new

    def solve(self):
        """Main optimization loop"""
        for _ in range(self.max_iter):
            # Store current solution
            self.history.append({
                'x': np.copy(self.x),
                'z': np.copy(self.z),
                'f_values': self.problem.evaluate(self.x, self.z)
            })
            
            # ADMM steps
            x_new = self._x_update()
            z_new = self._z_update()
            self._update_dual_variables(x_new, z_new)
            
            # Update variables
            self.x = x_new
            self.z = z_new
            
            # Adaptive regularization
            self.alpha *= self.beta
            
            # Check convergence
            if np.linalg.norm(x_new - z_new) < self.tol:
                break
                
        return self.x, self.z, self.problem.evaluate(self.x, self.z)

    def get_pareto_front(self):
        """Extract non-dominated solutions from history"""
        f_values = np.array([point['f_values'] for point in self.history])
        return f_values[NonDominatedSorting().do(f_values, only_non_dominated_front=True)]



class DUMMY1:
    def __init__(self, n=1, lb=-5, ub = 5, g_type = ('zero', {})):
        r"""
        JOS1 Problem
        f_1(x) = x**2
        f_2(x) = (x-2)**2

        g_1(z) = zero/L1/indicator/max
        g_2(z) = zero/L1/indicator/max

        x and z are kept different for the sake of generality over algorithms (mainly for ADMM, but in 
        ADMM also, we are looking for the cases where x=z, so constraints are defined mainly for x)


        Arguments:
        m: Number of objectives
        n: Number of variables
        ub: Upper bound for variables
        g_type: Type of g function
        It can be one of ('zero', 'L1', 'indicator', 'max')
        Its parameters must be defined in the dictionary

        Defined Vars:
        bounds: Bounds for variables [(low, high), ...]

        constraints: List of constraints for the problem, these must be defined as per scipy optimize
        like {'type': 'ineq', 'fun': lambda x: x[0] - 1}.
        """

        self.m = 2  # Number of objectives
        self.n = n
        self.lb = lb
        self.ub = ub
        self.bounds = [(lb, ub) for _ in range(n)] # Assuming 2 variables for JOS1
        self.constraints = []
        self.g_type = g_type
        self.true_pareto_front = self.calculate_optimal_pareto_front()
        # print(f"true_pareto_front: {self.true_pareto_front}")
        self.ref_point = np.max(self.true_pareto_front, axis=0)
        # print(f"ref_point: {self.ref_point}")

        # print(self.bounds)

    def calculate_optimal_pareto_front(self):
        """
        These are the values of the objectives at the true pareto front
        f_1(x) = 0.5 * \|x\|^2
        f_2(x) = 0.5 * \|x - 2\|^2
        """
        x = np.linspace(0, 2, 100)
        list_ = []
        for i in x:
            list_.append(self.evaluate(i, i))
        return np.array(list_)
    

    def f1(self, x):
        return x**2

    def grad_f1(self, x):
        return 2*x

    def f2(self, x):
        return (x - 2)**2

    def grad_f2(self, x):
        return 2*(x - 2)

    def hess_f1(self, x):
        return 2*np.eye(self.n)

    def hess_f2(self, x):
        return 2*np.eye(self.n)

    def evaluate_f(self, x):
        return np.array([self.f1(x), self.f2(x)])

    def evaluate_gradients_f(self, x):
        return np.array([self.grad_f1(x), self.grad_f2(x)])

    def evaluate_hessians_f(self, x):
        return np.array([self.hess_f1(x), self.hess_f2(x)])

    def evaluate_g(self, z):
        if self.g_type[0] == 'zero':
            return np.zeros(self.m)
        elif self.g_type[0] == 'L1':
            pass
        elif self.g_type[0] == 'indicator':
            pass
        else:
            pass

    def evaluate(self, x, z):
        f_values = self.evaluate_f(x)
        g_values = self.evaluate_g(z)
        return f_values + g_values    

    def f_list(self):
        return [
            (self.f1, self.grad_f1, self.hess_f1),
            (self.f2, self.grad_f2, self.hess_f2)
        ]

if __name__=="__main__":
    # Initialize problem and solver
    problem = DUMMY1()
    solver = MO_ADMM(problem, rho=1.0, alpha=0.5)

    # Solve with different initializations to get Pareto front
    pareto_points = []
    for _ in range(10):
        x_opt, z_opt, f_opt = solver.solve()
        print(f"x_opt = {x_opt}")
        print(f"z_opt = {z_opt}")
        print(f"f_opt = {f_opt}")
        pareto_points.append(problem.evaluate(x_opt, z_opt))
        solver.x = np.random.uniform(problem.lb, problem.ub, problem.n)  # Re-initialize

    # Extract non-dominated solutions
    final_front = NonDominatedSorting().do(np.array(pareto_points), only_non_dominated_front=True)