import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import os
style.use('ggplot')

def plot_pareto_front(Y_true, Y_approx, problem_name, save_path=None):
    """Plot true and approximated Pareto fronts"""
    plt.figure(figsize=(10, 6))
    plt.scatter(Y_true[:, 0], Y_true[:, 1], c='blue', label='True Pareto Front')
    plt.scatter(Y_approx[:, 0], Y_approx[:, 1], c='red', marker='x', label='Approximation')
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.title(f'Pareto Front Comparison {problem_name}')
    plt.legend()
    plt.grid(True)

    # if save_path:
    plt.savefig(save_path)
    plt.show()