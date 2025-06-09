import numpy as np
from scipy.spatial.distance import cdist
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def utililty(values, initial_points, max_pts=50, rank=2, convex=True):
    nds_indices = NonDominatedSorting().do(values)
    # print(f"nds_indices\n{nds_indices}")

    nd_values = []
    init_points = []
    for i in range(len(nds_indices)):
        if convex and i==rank:
            break
        nd_values.append(values[nds_indices[i]])
        init_points.append(initial_points[nds_indices[i]])

    nd_values = np.concatenate(nd_values)
    init_points = np.concatenate(init_points)
    # print(f"nd_values.shape = {nd_values.shape}, init_points.shape = {init_points.shape}")
    return nd_values[:max_pts], init_points[:max_pts]

def generational_distance(Y_N, Y_P, p=2):
    """
    Generational Distance (GD)

    Measures the average distance from each solution in the obtained front (Y_N) 
    to the closest solution in the true Pareto front (Y_P).

    Mathematically:
        GD = ( (1/n) * Σ_i d(y_i, Y_P)^p )^(1/p)
        where d(y_i, Y_P) = min_j ||y_i - y_j|| is the shortest Euclidean distance 
        from a point y_i in Y_N to the true front Y_P.

    Parameters:
    - Y_N : np.ndarray
        Obtained approximation set (solutions found).
    - Y_P : np.ndarray
        Reference Pareto front.
    - p : int (default=2)
        Power parameter for the Minkowski distance (p=2 gives Euclidean distance).

    Returns:
    - float : Generational Distance value.
    """
    D = cdist(Y_N, Y_P)
    min_dists = np.min(D, axis=1)
    return np.mean(min_dists**p)**(1/p)


def inverted_generational_distance(Y_N, Y_P, p=2):
    """
    Inverted Generational Distance (IGD)

    Measures the average distance from each point in the true Pareto front (Y_P) 
    to the nearest solution in the obtained front (Y_N).

    Mathematically:
        IGD = ( (1/m) * Σ_j d(y_j, Y_N)^p )^(1/p)

    Parameters:
    - Y_N : np.ndarray
        Obtained approximation set.
    - Y_P : np.ndarray
        Reference Pareto front.
    - p : int (default=2)
        Power for Minkowski distance.

    Returns:
    - float : Inverted Generational Distance value.
    """
    D = cdist(Y_P, Y_N)
    min_dists = np.min(D, axis=1)
    return np.mean(min_dists**p)**(1/p)


def hypervolume(Y_N, ref_point):
    """
    Hypervolume (HV)

    Measures the volume (in objective space) dominated by the obtained front (Y_N)
    with respect to a reference point.

    Mathematically:
        HV(Y_N, r) = volume( ∪_{y ∈ Y_N} [y, r] )

    Parameters:
    - Y_N : np.ndarray
        Set of non-dominated solutions.
    - ref_point : np.ndarray
        Reference point which should be dominated by all solutions in Y_N.

    Returns:
    - float : Hypervolume value.
    """
    ind = HV(ref_point=ref_point)
    return ind(Y_N)


def spacing(Y_N):
    """
    Spacing (SP)

    Measures the extent of uniform distribution (even spacing) among solutions 
    in the obtained front (Y_N).

    Mathematically:
        SP = sqrt( (1/n) * Σ_i (d_i - d̄)^2 )
        where d_i = min_j≠i ||y_i - y_j|| and d̄ is the mean of all d_i.

    Parameters:
    - Y_N : np.ndarray
        Obtained front of solutions.

    Returns:
    - float : Spacing value (lower is better).
    """
    D = cdist(Y_N, Y_N)
    np.fill_diagonal(D, np.inf)
    min_dists = np.min(D, axis=1)
    return np.sqrt(np.mean((min_dists - min_dists.mean())**2))


def spread(Y_N, Y_P):
    """
    Spread (Δ)

    Measures the extent of spread and distribution of solutions across the
    Pareto front, accounting for extreme points and spacing.

    Mathematically:
        Δ = (d_f + d_l + Σ_i |d̄ - d_i|) / (d_f + d_l + n*d̄)
        where:
          - d_f and d_l are distances from extreme points in Y_P to closest in Y_N
          - d_i is the distance between each point and its nearest neighbor
          - d̄ is the mean of d_i

    Parameters:
    - Y_N : np.ndarray
        Obtained front.
    - Y_P : np.ndarray
        True Pareto front (used to identify extreme points).

    Returns:
    - float : Spread value.
    """
    ideal = Y_P.min(axis=0)
    nadir = Y_P.max(axis=0)
    d_ext = np.linalg.norm(Y_N - ideal) + np.linalg.norm(Y_N - nadir)

    D = cdist(Y_N, Y_N)
    np.fill_diagonal(D, np.inf)
    d_i = np.min(D, axis=1)
    d_mean = d_i.mean()

    return (d_ext + np.abs(d_i - d_mean).sum()) / (d_ext + len(Y_N) * d_mean)


def epsilon_indicator(Y_N1, Y_N2):
    """
    Additive ε-indicator (Iε+)

    Measures the minimum value ε such that every point in Y_N2 is weakly dominated
    by at least one point in Y_N1 + ε.

    Mathematically:
        Iε+(Y_N1, Y_N2) = inf ε such that ∀y2 ∈ Y_N2, ∃y1 ∈ Y_N1: y1 + ε ≥ y2

    Parameters:
    - Y_N1 : np.ndarray
        First approximation front (usually the test/front being evaluated).
    - Y_N2 : np.ndarray
        Second approximation or reference front.

    Returns:
    - float : Additive ε-indicator value.
    """
    epsilons = []
    for y2 in Y_N2:
        min_eps = np.inf
        for y1 in Y_N1:
            eps = np.max(y1 - y2)
            if eps < min_eps:
                min_eps = eps
        epsilons.append(min_eps)
    return max(epsilons)


def coverage_metric(Y_N1, Y_N2):
    """
    Coverage Metric (C-Metric)

    Measures the fraction of solutions in Y_N2 that are dominated by at least
    one solution in Y_N1.

    Mathematically:
        C(Y_N1, Y_N2) = |{ y2 ∈ Y_N2 | ∃ y1 ∈ Y_N1: y1 ≤ y2 }| / |Y_N2|

    Parameters:
    - Y_N1 : np.ndarray
        First set (reference or better front).
    - Y_N2 : np.ndarray
        Second set to be compared against Y_N1.

    Returns:
    - float : Coverage value between 0 and 1.
    """
    count = 0
    for y2 in Y_N2:
        if np.any(np.all(Y_N1 <= y2, axis=1)):
            count += 1
    return count / len(Y_N2)


def r2_indicator(Y_N, weights, ref_point):
    """
    R2 Indicator

    Aggregates the quality of a solution set using a utility function based
    on weighted Tchebycheff scalarizing functions.

    Mathematically:
        R2(Y_N) = (1/|W|) * Σ_w∈W min_y∈Y_N max_i [ w_i * |y_i - r_i| ]

    Parameters:
    - Y_N : np.ndarray
        Obtained solution front.
    - weights : np.ndarray
        Set of uniformly distributed weight vectors.
    - ref_point : np.ndarray
        Reference point for scalarizing functions.

    Returns:
    - float : R2 quality indicator.
    """
    utilities = []
    for w in weights:
        weighted_dist = np.max(w * (Y_N - ref_point), axis=1)
        utilities.append(np.min(weighted_dist))
    return np.mean(utilities)


def averaged_hausdorff(Y_N, Y_P, p=2):
    """
    Averaged Hausdorff Distance (Δp)

    Combines both GD and IGD into a single metric by taking the max of both,
    penalizing poor convergence and diversity.

    Mathematically:
        Δp = max(GD_p(Y_N, Y_P), IGD_p(Y_N, Y_P))

    Parameters:
    - Y_N : np.ndarray
        Obtained front.
    - Y_P : np.ndarray
        True Pareto front.
    - p : int (default=2)
        Power parameter for Minkowski distance.

    Returns:
    - float : Averaged Hausdorff distance.
    """
    gd = generational_distance(Y_N, Y_P, p)
    igd = inverted_generational_distance(Y_N, Y_P, p)
    return max(gd, igd)


def error_ratio(Y_N, Y_P, tol=1e-6):
    """
    Error Ratio (ER)

    Measures the proportion of solutions in Y_N that are not present in Y_P 
    within a given tolerance.

    Mathematically:
        ER = |{ y ∈ Y_N | ∀y' ∈ Y_P, ||y - y'|| > tol }| / |Y_N|

    Parameters:
    - Y_N : np.ndarray
        Obtained front.
    - Y_P : np.ndarray
        True or reference Pareto front.
    - tol : float
        Tolerance threshold for equality (default: 1e-6).

    Returns:
    - float : Error ratio value (between 0 and 1).
    """
    count = 0
    for y in Y_N:
        if not np.any(np.all(np.isclose(y, Y_P, atol=tol), axis=1)):
            count += 1
    return count / len(Y_N)
