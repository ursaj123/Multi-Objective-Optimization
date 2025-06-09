import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D
import os
# rc("text", usetex=True)
# style.use(["science", "bright"])


def plot_2d_pareto_front(Y_true=None, Y_approx=None, initial_points = None, problem_name='', save_path=None, alpha=0.05):
    """Plot true and approximated Pareto fronts"""
    plt.figure(figsize=(10, 6))
    if Y_true is not None:
        plt.scatter(Y_true[:, 0], Y_true[:, 1], c='lightgrey', label='Feasible Space', s=10)
    if Y_approx is not None:
        plt.scatter(Y_approx[:, 0], Y_approx[:, 1], facecolor='green',marker='o', s=3, label='Approximate Pareto Front')

    if initial_points is not None and Y_approx is not None:
        plt.scatter(initial_points[:, 0], initial_points[:, 1], s=3, facecolor='cyan', marker='o', label='Initial Points')
        for start, end in zip(initial_points, Y_approx):
            plt.plot([start[0], end[0]], [start[1], end[1]], color='magenta', linewidth=0.5, alpha=0.6)


    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.title(f'Pareto Front {problem_name}')
    plt.legend()
    plt.grid(False)
    plt.show()

    if save_path:
        plt.savefig(save_path)
    plt.close()
    



def plot_3d_pareto_front(f_true=None, f_approx=None,
                        problem_name='',
                        elev=20, azim=-130,
                        initial_points=None,
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

    if f_approx is not None:
        ax.scatter(f_approx[:, 0], f_approx[:, 1], f_approx[:, 2],
           facecolor='green',marker='o', s=3, label='Approximate Pareto Front')


    if initial_points is not None and f_approx is not None:
        ax.scatter(initial_points[:, 0], initial_points[:, 1], initial_points[:, 2],
           s=3, facecolor='cyan', marker='o', label='Initial Points')
        for start, end in zip(initial_points, f_approx):
            xs, ys, zs = zip(start, end)
            ax.plot(xs, ys, zs, color='magenta', linewidth=0.5, alpha=0.6)

    if f_true is not None:
        ax.scatter(f_true[:, 0], f_true[:, 1], f_true[:, 2],
           c='lightgrey', label='Feasible Space', s=10)

    ax.set_xlabel('f1(x)')
    ax.set_ylabel('f2(x)')
    ax.set_zlabel('f3(x)')

    ax.view_init(elev=elev, azim=azim)
    ax.legend()
    ax.set_title(f'Pareto Front {problem_name}')
    plt.tight_layout()
    plt.grid(False)
    plt.show()
    if save_path:
        plt.savefig(save_path)

    plt.close(fig)