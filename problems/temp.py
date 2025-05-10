import numpy as np
from abc import ABC, abstractmethod
import math


class Foresnca:
    def __init__(self, m=2, n=2, bounds=None):
        super().__init__()
        self.m = m
        self.n = n
        self.bounds = None
        # self.bounds = tuple([(-5, 5) for _ in range(n)]) # Assuming 2 variables for JOS1
        # print(self.bounds)

    def f1(self, x):
        sum1 = sum((xi - 1 / np.sqrt(self.n)) ** 2 for xi in x)
        return 1 - np.exp(-sum1)

    def grad_f1(self, x):
        inv_sqrt_n = 1 / np.sqrt(self.n)
        sum1 = sum((xi - inv_sqrt_n) ** 2 for xi in x)
        return 2 * (x - inv_sqrt_n) * np.exp(-sum1)

    def f2(self, x):
        sum2 = sum((xi + 1 / np.sqrt(self.n)) ** 2 for xi in x)
        return 1 - np.exp(-sum2)

    def grad_f2(self, x):
        inv_sqrt_n = 1 / np.sqrt(self.n)
        sum2 = sum((xi + inv_sqrt_n) ** 2 for xi in x)
        return 2 * (x + inv_sqrt_n) * np.exp(-sum2)

    def f_list(self):
        return [
            (self.f1, self.grad_f1),
            (self.f2, self.grad_f2)
        ]


    
class JOS1:
    def __init__(self, m=2, n=2, bounds=None):
        super().__init__()
        self.m = m
        self.n = n
        self.bounds = None
        # self.bounds = tuple([(-5, 5) for _ in range(n)]) # Assuming 2 variables for JOS1
        # print(self.bounds)

    def f1(self, x):
        return 0.5 * np.sum(x**2)

    def grad_f1(self, x):
        return x

    def f2(self, x):
        return 0.5 * np.sum((x - 2)**2)

    def grad_f2(self, x):
        return x - 2

    def f_list(self):
        return [
            (self.f1, self.grad_f1),
            (self.f2, self.grad_f2)
        ]


class Poloni:
    def __init__(self, m=2, n=2, bounds=None):
        super().__init__()
        self.m = m
        self.n = n
        self.bounds = None

        # Precompute constants for the objective functions
        self.A = 0.5 * math.sin(1) - 2 * math.cos(1) + math.sin(2) - 1.5 * math.cos(2)
        self.C = 1.5 * math.sin(1) - math.cos(1) + 2 * math.sin(2) - 0.5 * math.cos(2)
        

    def f1(self, x):
        # Compute B and D for the given x and y
        B = 0.5 * math.sin(x[0]) - 2 * math.cos(x[0]) + math.sin(x[1]) - 1.5 * math.cos(x[1])
        D = 1.5 * math.sin(x[0]) - math.cos(x[0]) + 2 * math.sin(x[1]) - 0.5 * math.cos(x[1])
        
        # Objective f1
        f1 = 1 + (self.A - B)**2 + (self.C - D)**2
        return f1
        

    def f2(self, x):
        # Objective f2
        f2 = (x[0] + 3)**2 + (x[1] + 1)**2
        return f2

    def grad_f1(self, x):
        B = 0.5 * math.sin(x[0]) - 2 * math.cos(x[0]) + math.sin(x[1]) - 1.5 * math.cos(x[1])
        D = 1.5 * math.sin(x[0]) - math.cos(x[0]) + 2 * math.sin(x[1]) - 0.5 * math.cos(x[1])

        # Partial derivatives of B and D
        dB_dx = 0.5 * math.cos(x[0]) + 2 * math.sin(x[0])
        dB_dy = math.cos(x[1]) + 1.5 * math.sin(x[1])
        dD_dx = 1.5 * math.cos(x[0]) + math.sin(x[0])
        dD_dy = 2 * math.cos(x[1]) + 0.5 * math.sin(x[1])
        
        # Gradient of f1
        df1_dx = -2 * (self.A - B) * dB_dx - 2 * (self.C - D) * dD_dx
        df1_dy = -2 * (self.A - B) * dB_dy - 2 * (self.C - D) * dD_dy

        return np.array([df1_dx, df1_dy])

    def grad_f2(self, x):
        # Gradient of f2
        df2_dx = 2 * (x[0] + 3)
        df2_dy = 2 * (x[1] + 1)
        
        return np.array([df2_dx, df2_dy])

    def f_list(self):
        return [
            (self.f1, self.grad_f1),
            (self.f2, self.grad_f2)
        ]


class ZDT1:
    def __init__(self, m=2, n=30, bounds=(0, 1)):
        super().__init__()
        self.m = m
        self.n = n
        # self.bounds = None
        self.bounds = tuple([(bounds[0], bounds[1]) for _ in range(n)]) # Assuming 30 variables for ZDT1
        # print(self.bounds)

        pass

    def f1(self, x):
        return x[0]

    def f2(self, x):
        n = len(x)
        if n == 1:
            g = 1.0
        else:
            g = 1 + (9.0 / (n - 1)) * sum(x[1:])
        h = 1 - math.sqrt(x[0] / g)
        return g * h
    
    def jac_f1(self, x):
        return np.array([1.0] + [0.0] * (len(x) - 1))

    def jac_f2(self, x):
        n = len(x)
        if n == 1:
            g = 1.0
        else:
            g = 1 + (9.0 / (n - 1)) * sum(x[1:])
        
        grad_f2 = np.zeros(n)
        
        # Compute gradient of f2 for x1 (index 0)
        if x[0] == 0:
            grad_f2[0] = -np.inf
        else:
            grad_f2[0] = - (math.sqrt(g) / (2 * math.sqrt(x[0])))
        
        # Compute gradient of f2 for xi (i >= 2)
        if n > 1:
            term = math.sqrt(x[0] / g)
            factor = (9.0 / (n - 1)) * (1 - term / 2)
            for i in range(1, n):
                grad_f2[i] = factor
        
        return grad_f2


    def f_list(self):
        return [
            (self.f1, self.jac_f1),
            (self.f2, self.jac_f2)
        ]



class ZDT2:
    def __init__(self, m=2, n=30, bounds=(0, 1)):
        super().__init__()
        self.m = m
        self.n = n
        self.bounds = tuple([(bounds[0], bounds[1]) for _ in range(n)]) # Assuming 30 variables for ZDT1
        pass

    def f1(self, x):
        return x[0]

    def f2(self, x):
        n = len(x)
        if n == 1:
            g = 1.0
        else:
            g = 1 + (9.0 / (n - 1)) * sum(x[1:])
        h = 1 - (x[0] / g)**2
        return g * h

    def jac_f1(self, x):
        return np.array([1.0] + [0.0] * (len(x) - 1))

    def jac_f2(self, x):
        n = len(x)
        if n == 1:
            g = 1.0
        else:
            g = 1 + (9.0 / (n - 1)) * sum(x[1:])
        
        grad_f2 = [0.0] * n
        grad_f2[0] = -2 * x[0] / g  # Gradient of f2 w.r.t. x1
        
        if n > 1:
            term = (9.0 / (n - 1)) * (1 + (x[0] ** 2) / (g ** 2))
            for i in range(1, n):
                grad_f2[i] = term  # Gradient of f2 for x2 to xn
    
        return np.array(grad_f2)

    def f_list(self):
        return [
            (self.f1, self.jac_f1),
            (self.f2, self.jac_f2)
        ]
    


class Rosenbrock:
    def __init__(self, m=1, n=2, bounds=None):
        super().__init__()
        self.m = m
        self.n = n
        self.bounds = None

    def f1(self, x):
        return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

    def grad_f1(self, x):
        df1_dx = np.array([-400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]), 200 * (x[1] - x[0]**2)])
        return df1_dx

    def f_list(self):
        return [
            (self.f1, self.grad_f1)
        ]



class Easom:
    def __init__(self, m=1, n=2, bounds=None):
        super().__init__()
        self.m = m
        self.n = n
        self.bounds = None

    def f1(self, x):
        return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-(x[0] - np.pi)**2 - (x[1] - np.pi)**2)
        
    def grad_f1(self, x):
        delta_x = x[0] - np.pi
        delta_y = x[1] - np.pi
        exp_term = np.exp(-delta_x**2 - delta_y**2)
        cos_x = np.cos(x[0])
        cos_y = np.cos(x[1])
        sin_x = np.sin(x[0])
        sin_y = np.sin(x[1])
        
        df_dx = cos_y * exp_term * (sin_x + 2 * delta_x * cos_x)
        df_dy = cos_x * exp_term * (sin_y + 2 * delta_y * cos_y)

        return np.array([df_dx, df_dy])
        

    def f_list(self):
        return [
            (self.f1, self.grad_f1)
        ]
    

class Circular:
    def __init__(self, m=1, n=2, bounds=None):
        super().__init__()
        self.m = m
        self.n = n
        self.bounds = None
        cond_num =  1e2
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





        