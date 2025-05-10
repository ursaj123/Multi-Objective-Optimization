## 1. Single Objective ADMM
The single objective ADMM problem is given by:

$min_{x, z}$ $f(x) + g(z)$

$\text{s.t.}$ $Ax + Bz = c$, $x \in \mathbb{R^n}$

The augmented lagrangian for the i'th component is given by:

>$L_{\rho}(x, z, \lambda) = f(x) + g(z) + \lambda^T(Ax+Bz-c) + \frac{\rho}{2}\|Ax + Bz-c\|_2^2$

The ADMM algo for single objective optimization is given by

>$x^{k+1} = argmin_x [f(x) + \lambda^T(Ax+Bz-c) + \frac{\rho}{2}\|Ax + Bz-c\|_2^2]$

>$z^{k+1} = argmin_z [g(z) + \lambda^T(Ax+Bz-c) + \frac{\rho}{2}\|Ax + Bz-c\|_2^2]$

>$\lambda^{k+1} = \lambda^k + \rho(Ax^{k+1} + Bz^{k+1} - c)$

## 2. Multi-Objective ADMM (Maximization)
The multi-objective ADMM problem is given by:

$min_{x, z}$ $F(x) + G(z)$

$\text{s.t.}$ $Ax + Bz = c$, $x \in \mathbb{R^n}$

where $F(x) = [f_1, f_2, ..., f_m]^T$ and $G(z) = [g_1, g_2, ..., g_m]^T$


The augmented lagrangian for the i'th component is given by:

>$(L_{\rho}(x, z, \lambda))_i = f_i(x) + g_i(z) + \lambda^T(Ax+Bz-c) + \frac{\rho}{2}\|Ax + Bz-c\|_2^2$

The new proposed algorithm is given by

>$x^{k+1} = argmin_x [(max_{i = 1,2,...,m} f_i(x)) + \lambda^T(Ax+Bz-c) + \frac{\rho}{2}\|Ax + Bz-c\|_2^2]$

>$z^{k+1} = argmin_z [(max_{i = 1,2,...,m} g_i(z)) + \lambda^T(Ax+Bz-c) + \frac{\rho}{2}\|Ax + Bz-c\|_2^2]$

>$\lambda^{k+1} = \lambda^k + \rho(Ax^{k+1} + Bz^{k+1} - c)$


