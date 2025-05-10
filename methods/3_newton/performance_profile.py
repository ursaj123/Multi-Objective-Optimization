import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


def profile(solvers_metrics, solvers_names, metric_name = 'time_elapsed', data_dir = 'newton_results', file_name = 'performance_profile.png', n_pts=100):
    r"""
    solvers_metrics is an array of size (s, p)
    where s is the number of solvers and p is the number of metrics

    and the metric could be anything like time, hypervolume, spread metrics whatever.
    """
    print(f"solvers_metrics: {solvers_metrics}")
    s, p = solvers_metrics.shape
    rm_max_along_columns = np.max(solvers_metrics, axis=0)
    rm_max_along_columns = 1

    print(f"r_p_s = {solvers_metrics}")
    r_p_s = (np.array(solvers_metrics).copy())/rm_max_along_columns
    print(f"r_p_s: {r_p_s}")

    # figure
    fig = plt.figure(figsize=(10, 6))
    plt.title('Performance Profile')
    for i in range(s):
        ys = []
        xs = np.linspace(min(0, np.min(r_p_s[i])), max(1, np.max(r_p_s[i])), n_pts)
        print(f"xs: {xs}")
        
        for j in xs:
            ratio = np.sum(r_p_s[i]<=j)/p
            ys.append(ratio)
        
        ys = np.array(ys)
        plt.plot(xs, ys, label=solvers_names[i], linewidth=2)
        plt.xlabel(f"{metric_name}")
        plt.ylabel('rho_s(tau)')
        
    plt.legend()
    plt.show()
    plt.savefig(f'{data_dir}/{file_name}')
    pass


