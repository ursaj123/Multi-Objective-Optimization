import numpy as np

class FA1:
    """
    FA1 test-problem with three objectives and three decision variables.

    f1(x1) = (1 - exp(-4 x1)) / (1 - exp(-4))
    f2(x)  = (x2 + 1) * (1 - sqrt(f1(x1) / (x2 + 1)))
    f3(x)  = (x3 + 1) * (1 - (f1(x1) / (x3 + 1))**0.1)

    Optional additive g-terms (zero, L1, …) can be switched on through g_type.
    """

    # ---------- construction -------------------------------------------------
    def __init__(self, g_type=('zero', {}), fact=1):
        self.m = 3            # objectives
        self.n = 3            # variables
        self.bounds = [(0, 1), (-5, 5), (-5, 5)]
        self.constraints = []
        self.g_type = g_type

        # set-up for the optional L1 g-terms
        self.l1_ratios, self.l1_shifts = [], []
        if self.g_type[0] == 'L1':
            for i in range(self.m):
                self.l1_ratios.append(1 / ((i + 1) * self.n*fact))
                self.l1_shifts.append(i)
            self.l1_ratios = np.asarray(self.l1_ratios)
            self.l1_shifts = np.asarray(self.l1_shifts)

    # ---------- helper: finite-difference Hessian ----------------------------

    def feasible_space(self):
        test_x = []
        for i in range(self.n):
            low, high = self.bounds[i]
            test_x.append(np.random.uniform(low, high, size=(50000,)))
        test_x = np.array(test_x).T  # Shape (1000, n) 
        f_values = np.array([self.evaluate(x, x) for x in test_x])
        return f_values
    
    def _fd_hessian(self, grad_fun, x, eps=1e-8):
        """Symmetric centred-difference Hessian."""
        n = len(x)
        hess = np.zeros((n, n))
        base_grad = grad_fun(x)
        for j in range(n):
            x_fwd = np.array(x, copy=True);  x_fwd[j] += eps
            x_bwd = np.array(x, copy=True);  x_bwd[j] -= eps
            grad_fwd = grad_fun(x_fwd)
            grad_bwd = grad_fun(x_bwd)
            # ∂²f / ∂x_i∂x_j  ≈ (∇f(x+e_j) - ∇f(x-e_j)) / (2ε)
            hess[:, j] = (grad_fwd - grad_bwd) / (2.0 * eps)
        # enforce exact symmetry
        return 0.5 * (hess + hess.T)

    # ---------- f1 -----------------------------------------------------------
    def f1(self, x):
        return (1.0 - np.exp(-4.0 * x[0])) / (1.0 - np.exp(-4.0))

    def grad_f1(self, x):
        g = np.zeros(self.n)
        g[0] = 4.0 * np.exp(-4.0 * x[0]) / (1.0 - np.exp(-4.0))
        return g

    def hess_f1(self, x):
        h = np.zeros((self.n, self.n))
        h[0, 0] = -16.0 * np.exp(-4.0 * x[0]) / (1.0 - np.exp(-4.0))
        return h

    # ---------- f2 -----------------------------------------------------------
    def f2(self, x):
        A  = x[1] + 1.0                      # A := x2 + 1
        y  = self.f1(x)                      # y := f1(x1)
        return A * (1.0 - np.sqrt(y / A))

    def grad_f2(self, x):
        y  = self.f1(x)
        dy = self.grad_f1(x)[0]              # scalar dy/dx1
        A  = x[1] + 1.0
        B  = y / A                           # convenience

        g  = np.zeros(self.n)
        # ∂/∂x1
        g[0] = -0.5 * dy / np.sqrt(B)
        # ∂/∂x2
        g[1] = 1.0 - 0.5 * np.sqrt(B)
        # ∂/∂x3
        g[2] = 0.0
        return g

    def hess_f2(self, x):
        return self._fd_hessian(self.grad_f2, x)

    # ---------- f3 -----------------------------------------------------------
    def f3(self, x):
        C  = x[2] + 1.0
        y  = self.f1(x)
        return C * (1.0 - (y / C) ** 0.1)

    def grad_f3(self, x):
        y  = self.f1(x)
        dy = self.grad_f1(x)[0]              # scalar dy/dx1
        C  = x[2] + 1.0
        B  = y / C

        g  = np.zeros(self.n)
        # ∂/∂x1
        g[0] = -0.1 * dy * B ** (-0.9)
        # ∂/∂x2
        g[1] = 0.0
        # ∂/∂x3
        g[2] = 1.0 - 0.9 * B ** 0.1
        return g

    def hess_f3(self, x):
        return self._fd_hessian(self.grad_f3, x)

    # ---------- convenience evaluators --------------------------------------
    def evaluate_f(self, x):
        return np.array([self.f1(x), self.f2(x), self.f3(x)])

    def evaluate_gradients_f(self, x):
        return np.array([self.grad_f1(x),
                         self.grad_f2(x),
                         self.grad_f3(x)])

    def evaluate_hessians_f(self, x):
        return np.array([self.hess_f1(x),
                         self.hess_f2(x),
                         self.hess_f3(x)])

    # ---------- optional g-terms --------------------------------------------
    def evaluate_g(self, z):
        if self.g_type[0] == 'zero':
            return np.zeros(self.m)

        if self.g_type[0] == 'L1':
            res = np.zeros(self.m)
            for i in range(self.m):
                res[i] = np.linalg.norm(
                    (z - self.l1_shifts[i]) * self.l1_ratios[i], 1
                )
            return res

        # indicator / max: not yet implemented
        return np.zeros(self.m)

    # total objective vector --------------------------------------------------
    def evaluate(self, x, z):
        return self.evaluate_f(x) + self.evaluate_g(z)

    # list-like interface -----------------------------------------------------
    def f_list(self):
        return [
            (self.f1, self.grad_f1, self.hess_f1),
            (self.f2, self.grad_f2, self.hess_f2),
            (self.f3, self.grad_f3, self.hess_f3)
        ]
