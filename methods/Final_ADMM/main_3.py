# Initialize the problem and algorithm
# PYTHONWARNINGS="ignore::UserWarning" python methods/Final_ADMM/main.py

import os
import sys
import numpy as np
import pandas as pd
from itertools import product
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from joblib import Parallel, delayed

import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join('/'.join(os.path.dirname(__file__).split('/')[:-2]), 'metrics'))
sys.path.append(os.path.join('/'.join(os.path.dirname(__file__).split('/')[:-2]), 'problems'))
sys.path.append(os.path.join('/'.join(os.path.dirname(__file__).split('/')[:-2]), 'plotting'))

from problems import class_list
from MOP_metrics import *
from moadmm import MultiObjectiveADMM
from pareto_front_plots import plot_2d_pareto_front, plot_3d_pareto_front


print(f"Problem list - {class_list.keys()}")
print('Modules imported successfully')

os.makedirs(os.path.join(os.path.dirname(__file__), 'results_new'), exist_ok=True)



# problem = class_list['JOS1'](n=2, g_type=('L1', {}))
# test_x = np.random.uniform(problem.bounds[0][0], problem.bounds[0][1], size=(10000, 2))
# f_values = np.array([problem.evaluate_f(x) for x in test_x])
# # plot_3d_pareto_front(f_values, azim=-93, alpha=0.1, save_path='tpf.png')
# plot_2d_pareto_front(problem.feasible_space(), alpha=0.1, save_path='tpf.png')
# sys.exit()





#############################  BASIC TEST CASE HANDLING #######################################
# let us first confirm that the algorithm is well implemented (basic test case, where m=1
# we will be comparing the approximated answer of CIRCULAR problem with the original soln, if it lies in proximity,
# then only we will proceed for the multiobjective optimization

# N, COND_NUM = [20, 30], [1e1, 1e2, 1e3]
# # N, COND_NUM = [10], [1e2]
# for n, cond_num in product(N, COND_NUM):
#     print(f"{'-'*50}")
#     print(f"Problem - CIRCULAR, n = {n}, cond_num = {cond_num}")
#     problem = class_list['CIRCULAR'](n=n, cond_num=cond_num)
#     solver = MultiObjectiveADMM(problem)
#     result = solver.solve()
#     print(f"result = {result}")
#     print(f"Result Difference = {np.linalg.norm(result['x'] - problem.orig_soln)}")

#     print(f"{'-'*50}")


# # great, it works
# sys.exit()
##################################################################################################





def calculate_metrics(Y_N, Y_P, ref_point=None, num_samples = 50):
    """Calculate all metrics for approximation set"""
    if ref_point is None:
        ref_point = np.max(Y_P, axis=0) + 1e-6
    
    weights = np.random.dirichlet(np.ones(Y_P.shape[1]), num_samples)
    
    return {
        'GD': generational_distance(Y_N, Y_P),
        'IGD': inverted_generational_distance(Y_N, Y_P),
        'Hypervolume': hypervolume(Y_N, ref_point),
        'Spacing': spacing(Y_N),
        'Spread': spread(Y_N, Y_P),
        'Epsilon': epsilon_indicator(Y_N, Y_P),
        'Coverage': coverage_metric(Y_N, Y_P),
        'R2': r2_indicator(Y_N, weights, ref_point),
        'Average_Hausdroff': averaged_hausdorff(Y_N, Y_P),
        'Error_Ratio': error_ratio(Y_N, Y_P)
    }

def initialize_x(problem, num_samples=100):
    # print(problem.bounds)
    initial_points = []
    for i in range(num_samples):
        x0 = np.random.uniform(-1.0, 1.0, problem.n)
        if problem.bounds is not None:
            x0 = np.array([np.random.uniform(lb, ub) for lb, ub in problem.bounds])
        z0 = x0.copy()
        initial_points.append((x0, z0))
    return initial_points


problem_lists = ['MOP2', 'SD', 'TOI4', 'TRIDIA']
convex_problems = {'AP1', 'AP2', 'AP4', 'BK1', 'DGO2', 'FDS', 'IKK1', 'JOS1', 'LOV1', 'SD', 'TOI4', 'TRIDIA'}
num_samples = 150
max_outer = 200
max_inner = 20
rho = 1.0

failed = []

def run(solver, start_points, n_jobs=-1):
    """Run the solver in parallel for different initial points."""
    results = Parallel(n_jobs=n_jobs)(delayed(solver.solve)(x0, z0) for x0, z0 in start_points)
    return results


# keeping norm_constrained only [''] instead of ['', 'Linf']
for problem_key, g_type, norm_constraint in product(problem_lists, [('zero', {}), ('L1', {})], ['']):
    try:
        problem = class_list[problem_key](g_type=g_type, fact=3)
        # problem_name = f'{problem_key}, g = {g_type[0]}, norm constraint = {norm_constraint}'
        problem_name = f'{problem_key}, g = {g_type[0]}'
        print(problem_name)

        solver = MultiObjectiveADMM(problem=problem, max_inner=max_inner, max_outer=max_outer, norm_constraint=norm_constraint, g_type=problem.g_type, rho=rho)
        initial_points = initialize_x(problem, num_samples)
        run_results = run(solver, initial_points, n_jobs=-1)


        results, graph_init = [], []
        for i, res in enumerate(run_results):
            if np.linalg.norm(res['x']-res['z'])<=1e-1:
                results.append(res)
                graph_init.append(initial_points[i])
            
        # plotting pareto front
        if problem_key in convex_problems:
            Y_approx, graph_init = utililty(np.array([result['f'] for result in results]), np.array(graph_init), convex=True)
        else:
            Y_approx, graph_init = utililty(np.array([result['f'] for result in results]), np.array(graph_init), convex=False)
        
        if problem.m==2:
            plot_2d_pareto_front(
                # Y_true=problem.true_pareto_front,
                Y_true=problem.feasible_space(),
                Y_approx=Y_approx,
                # initial_points=np.array([result['intial_points'] for result in results]),
                initial_points = np.array([problem.evaluate(init[0], init[1]) for init in graph_init]),
                problem_name=problem_name, # later append all of the settings to the string also
                save_path=os.path.join(os.path.dirname(__file__), "results_new", f"{problem_name}_pareto_front.png")
            )
        elif problem.m==3:
            plot_3d_pareto_front(
                # f_true=problem.true_pareto_front,
                # f_true=problem.feasible_space(),
                f_approx=Y_approx,
                # initial_points=np.array([result['intial_points'] for result in results]),
                initial_points = np.array([problem.evaluate(init[0], init[1]) for init in graph_init]),
                problem_name=problem_name, # later append all of the settings to the string also
                save_path=os.path.join(os.path.dirname(__file__), "results_new", f"{problem_name}_pareto_front.png")
            )



        # Convert results to DataFrame
        Y_N = np.array([result['f'] for result in results])
        # print(Y_N)
        Y_P = Y_N[NonDominatedSorting().do(Y_N, only_non_dominated_front=True)]
        ref_point = np.max(Y_P, axis=0) + 1e-6
        # print(f"Y_N shape: {Y_N.shape}, Y_P shape: {Y_P.shape}")
        metrics_res = calculate_metrics(
            Y_N = Y_N,
            Y_P = Y_P,
            ref_point = ref_point,
            num_samples = num_samples
        )
        
        metrics_res['avg_time'] = np.mean([result['runtime'] for result in results])
        metrics_res['avg_iter'] = np.mean([result['iterations'] for result in results])
        metrics_res['problem'] = problem_name


        # let us first convert everything to 4 decimal points
        for key in metrics_res.keys():
            if key not in ['problem']:
                metrics_res[key] = np.round(metrics_res[key], 4)

        # let us create a dataframe for the metrics 
        if not os.path.exists(os.path.join(os.path.dirname(__file__), 'results_new', 'metrics.csv')):
            # create a file
            metrics_df = pd.DataFrame(
                columns=[
                    'problem', 'avg_time', 'avg_iter', 'GD', 'IGD', 'Hypervolume', 'Spacing', 'Spread',
                    'Epsilon', 'Coverage', 'R2', 'Average_Hausdroff', 'Error_Ratio'
                ]
            )
            metrics_df.to_csv(os.path.join(os.path.dirname(__file__), 'results_new', 'metrics.csv'), index=False)


        # let us read the csv file and then concat to it
        metrics_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'results_new', 'metrics.csv'))
        metrics_df = pd.concat([metrics_df, pd.DataFrame(metrics_res, index=[0])], ignore_index=True)
        metrics_df.to_csv(os.path.join(os.path.dirname(__file__), 'results_new', 'metrics.csv'), index=False)

        print(metrics_res)

    except:
        # let us also print the traceback
        import traceback
        traceback.print_exc()
        print(f"Error occurred for problem {problem_name}, g_type = {g_type}, norm_constraint = {norm_constraint}")
        failed.append((problem_name, g_type, norm_constraint))

print(f"Failed problems: {failed}")
