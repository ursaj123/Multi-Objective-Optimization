import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D
import os
style.use('ggplot')

def plot_2d_pareto_front(Y_true=None, Y_approx=None, problem_name='', save_path=None):
    """Plot true and approximated Pareto fronts"""
    plt.figure(figsize=(10, 6))
    if Y_true is not None:
        plt.scatter(Y_true[:, 0], Y_true[:, 1], c='blue', label='True Pareto Front')
    if Y_approx is not None:
        plt.scatter(Y_approx[:, 0], Y_approx[:, 1], c='red', marker='x', label='Approximation')

    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.title(f'Pareto Front Comparison {problem_name}')
    plt.legend()
    plt.grid(True)
    plt.show()

    if save_path:
        plt.savefig(save_path)
    plt.close()
    



def plot_3d_pareto_front(f_true=None, f_approx=None,
                          problem_name='',
                          elev=20, azim=-93,
                          color_true='black', color_approx='red',
                          label_true='True', label_approx='Approx',
                          point_size=5,
                          save_path=None):
    """
    Plots two 3D scatter plots in the same figure with custom orientation.

    Parameters:
    - f1_true, f2_true, f3_true: Arrays for the true values
    - f1_approx, f2_approx, f3_approx: Arrays for the approximate values
    - elev: Elevation angle for the view (default: 20)
    - azim: Azimuth angle for the view (default: -135)
    - color_true: Color of the true points (default: 'black')
    - color_approx: Color of the approximate points (default: 'red')
    - label_true: Label for the true points
    - label_approx: Label for the approximate points
    - point_size: Size of the scatter points (default: 5)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if f_true is not None:
        ax.scatter(f_true[:, 0], f_true[:, 1], f_true[:, 2], color=color_true, s=point_size, label=label_true)

    if f_approx is not None:
        ax.scatter(f_approx[:, 0], f_approx[:, 1], f_approx[:, 2], color=color_approx, s=point_size, label=label_approx)

    ax.set_xlabel(r'$f_1(x)$')
    ax.set_ylabel(r'$f_2(x)$')
    ax.set_zlabel(r'$f_3(x)$')

    ax.view_init(elev=elev, azim=azim)
    ax.legend()
    ax.set_title(f'Pareto Front Comparison {problem_name}')
    plt.tight_layout()
    plt.show()
    if save_path:
        plt.savefig(save_path)

    plt.close(fig)