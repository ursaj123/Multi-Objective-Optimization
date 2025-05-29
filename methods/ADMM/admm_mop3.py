import numpy as np
from scipy.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

class MO_ADMM:
    def __init__(self, problem, rho=1.0, alpha=0.1, max_iter=100):
        self.problem = problem
        self.rho = rho
        self.alpha = alpha
        self.max_iter = max_iter
        self.history = []
        
        # Initialize with parameterized dual variables
        self.u = [np.random.uniform(-2, 2, problem.n) for _ in range(problem.m)]
        self.x = np.random.uniform(problem.lb, problem.ub, problem.n)
        self.z = np.copy(self.x)

    def _x_update(self):
        """Pareto-optimal x-update with proper epigraph formulation"""
        n = self.problem.n
        def obj_func(vars):
            return vars[n]  # Minimize t

        constraints = []
        for i in range(self.problem.m):
            f, grad_f, _ = self.problem.f_list()[i]
            def con_func(vars, i=i):
                x = vars[:n]
                return vars[n] - (
                    f(x) + 
                    self.rho/2 * np.linalg.norm(x - self.z + self.u[i])**2 +
                    self.alpha/2 * np.linalg.norm(x - self.x)**2
                )
            constraints.append({'type': 'ineq', 'fun': con_func})

        res = minimize(obj_func,
                       x0=np.concatenate([self.x, [0]]),
                       constraints=constraints,
                       bounds=self.problem.bounds + [(None, None)])
        return res.x[:n]

    def _z_update(self):
        """Enhanced z-update with explicit Pareto tradeoff control"""
        if self.problem.g_type[0] == 'zero':
            # Weighted average based on dual variables
            weights = [np.linalg.norm(ui) for ui in self.u]
            weights = np.array(weights) / sum(weights)
            return sum(w*(self.x + ui) for w, ui in zip(weights, self.u))
        
        # Other g_type handlers remain the same...

    def solve(self, tol=1e-6):
        for _ in range(self.max_iter):
            x_new = self._x_update()
            z_new = self._z_update()
            
            # Store all intermediate solutions
            self.history.append({
                'x': x_new,
                'f_values': self.problem.evaluate(x_new, z_new)
            })
            
            # Update dual variables with relaxation
            for i in range(self.problem.m):
                self.u[i] += 0.7 * (x_new - z_new)  # Relaxation factor
                
            self.x = x_new
            self.z = z_new

        return self.x, self.z

    def generate_pareto_front(self, n_runs=20):
        """Systematic Pareto front generation"""
        solutions = []
        
        for _ in range(n_runs):
            # Perturb dual variables to explore different tradeoffs
            self.u = [np.random.uniform(-2, 2, self.problem.n) 
                     for _ in range(self.problem.m)]
            self.solve()
            solutions.append(self.history[-1]['f_values'])

        print("Generated Solutions:", solutions)
            
        # Filter non-dominated solutions
        return np.array(solutions)[NonDominatedSorting().do(np.array(solutions))]




import numpy as np

class JOS1:
    def __init__(self, n=2, lb=-5, ub = 5, g_type = ('zero', {})):
        r"""
        JOS1 Problem
        f_1(x) = 0.5 * \|x\|^2
        f_2(x) = 0.5 * \|x - 2\|^2

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
        self.ref_point = np.max(self.true_pareto_front, axis=0) + 1e-4

        # print(self.bounds)

    def calculate_optimal_pareto_front(self):
        """
        These are the values of the objectives at the true pareto front
        f_1(x) = 0.5 * \|x\|^2
        f_2(x) = 0.5 * \|x - 2\|^2
        """
        return np.array([[1.0, 1.0], [4.0, 0.0], [0.0, 4.0]])
    

    def f1(self, x):
        return 0.5 * np.sum(x**2)

    def grad_f1(self, x):
        return x

    def f2(self, x):
        return 0.5 * np.sum((x - 2)**2)

    def grad_f2(self, x):
        return x - 2

    def hess_f1(self, x):
        return np.eye(self.n)

    def hess_f2(self, x):
        return np.eye(self.n)

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
    problem = JOS1(n=2, g_type=('zero', {}))
    solver = MO_ADMM(problem)
    pareto_front = solver.generate_pareto_front()

    print("Computed Pareto Front:")
    print(pareto_front)