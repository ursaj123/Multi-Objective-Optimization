# Initialize the problem and algorithm
import os
import sys
import numpy as np
import pandas as pd
from itertools import product

sys.path.append(os.path.join('/'.join(os.path.dirname(__file__).split('/')[:-2]), 'metrics'))
sys.path.append(os.path.join('/'.join(os.path.dirname(__file__).split('/')[:-2]), 'problems'))
sys.path.append(os.path.join('/'.join(os.path.dirname(__file__).split('/')[:-2]), 'plotting'))

from problems import class_list
from MOP_metrics import *
from moadmm import MultiObjectiveADMM
from pareto_front_plots import plot_2d_pareto_front, plot_3d_pareto_front


print(f"Problem list - {class_list.keys()}")
print('Modules imported successfully')

os.makedirs(os.path.join(os.path.dirname(__file__), 'results'), exist_ok=True)




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

# problem_lists = ['JOS1', 'SD', 'ZDT1', 'TOI4', 'TRIDIA', 'FDS']
# problem_lists = ['JOS1', 'SD', 'ZDT1']
problem_lists = ['POLONI'] # n = [5, 10, 20, 30, 50, 100]
# dimension = [2]
# dimension = [5, 10, 100]
dimension = [2]


# for key in class_list.keys():
for problem_key in problem_lists:
    for n in dimension:
        # if key=='CIRCULAR':
        #     continue
        # if key!="MOP5":
        #     continue
        # print(f"Problem - {key}")

        num_samples = 100
        max_outer = 200
        max_inner = 20
        rho = 1.0
        norm_constraint = 'Linf'
        # norm_constraint = ''
        g_type = ('L1', {})
        # g_type = ('zero', {})
        problem = class_list[problem_key](n=n, g_type=g_type)
        problem_name = f'{problem_key}_with_norm_constraint_Linf_g_type_{g_type}_n={n}'
        print(problem_name)

        # basic checking  if problem functions and its gradients are working well
        # x = np.ones(n)*0.5
        # print(f"f_val: {problem.evaluate_f(x)}, g_val: {problem.evaluate_g(x)}")
        # print(f"f: {problem.evaluate(x, x)}")
        # print(f"grad: {problem.evaluate_gradients_f(x)}")
        # sys.exit()
        
        # print(f"Problem - {key}, n = {n}")
        # continue
        # problem_name = problem.__class__.__name__ + f"_n={n}"
        
        # print(problem.__class__.__name__)

        solver = MultiObjectiveADMM(problem=problem, max_inner=max_inner, max_outer=max_outer, norm_constraint=norm_constraint, g_type=problem.g_type, rho=rho)
        results = []
        for i in range(num_samples):
            print(f"{'-'*20}Sample {i+1}/{num_samples}{'-'*20}")
            result = solver.solve()
            print(result)
            results.append(result)
            print(f"{'-'*50}")

        print(os.path.join(os.path.dirname(__file__), "results", f"{problem_name}_pareto_front.png"))
        # sys.exit()

        # plotting pareto front
        if problem.m==2:
            plot_2d_pareto_front(
                # Y_true=problem.true_pareto_front,
                Y_true=None,
                Y_approx=np.array([result['f_val'] for result in results]),
                problem_name=problem_name, # later append all of the settings to the string also
                save_path=os.path.join(os.path.dirname(__file__), "results", f"{problem_name}_pareto_front.png")
            )
        elif problem.m==3:
            plot_3d_pareto_front(
                # f_true=problem.true_pareto_front,
                f_true=None,
                f_approx=np.array([result['f'] for result in results]),
                problem_name=problem_name, # later append all of the settings to the string also
                save_path=os.path.join(os.path.dirname(__file__), "results", f"{problem_name}_pareto_front.png")
            )



        # Convert results to DataFrame
        metrics_res = calculate_metrics(
            Y_N = np.array([result['f'] for result in results]),
            Y_P = problem.true_pareto_front,
            ref_point = problem.ref_point,
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
        if not os.path.exists(os.path.join(os.path.dirname(__file__), 'results', 'metrics.csv')):
            # create a file
            metrics_df = pd.DataFrame(
                columns=[
                    'problem', 'avg_time', 'avg_iter', 'GD', 'IGD', 'Hypervolume', 'Spacing', 'Spread',
                    'Epsilon', 'Coverage', 'R2', 'Average_Hausdroff', 'Error_Ratio'
                ]
            )
            metrics_df.to_csv(os.path.join(os.path.dirname(__file__), 'results', 'metrics.csv'), index=False)


        # let us read the csv file and then concat to it
        metrics_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'results', 'metrics.csv'))
        metrics_df = pd.concat([metrics_df, pd.DataFrame(metrics_res, index=[0])], ignore_index=True)
        metrics_df.to_csv(os.path.join(os.path.dirname(__file__), 'results', 'metrics.csv'), index=False)

        print(metrics_res)
