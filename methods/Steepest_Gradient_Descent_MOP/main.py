# Initialize the problem and algorithm
import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.join('/'.join(os.path.dirname(__file__).split('/')[:-2]), 'metrics'))
sys.path.append(os.path.join('/'.join(os.path.dirname(__file__).split('/')[:-2]), 'problems'))
sys.path.append(os.path.join('/'.join(os.path.dirname(__file__).split('/')[:-2]), 'plotting'))

from problems import class_list
from MOP_metrics import *
from steepest_grad_desc import MultiObjectiveSteepestDescent
from pareto_front_plots import plot_pareto_front


print(f"Problem list - {class_list.keys()}")
print('Modules imported successfully')

os.makedirs(os.path.join(os.path.dirname(__file__), 'results'), exist_ok=True)



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

for key in class_list.keys():
    print(f"Problem - {key}")

    num_samples = 50
    problem = class_list[key]()
    # print(problem.__class__.__name__)

    solver = MultiObjectiveSteepestDescent(
        problem=problem,
        beta=0.1,
        sigma=0.1,
        tol=1e-4,
        max_iter=100,
    )
    results = []
    for i in range(num_samples):
        print(f"{'-'*20}Sample {i+1}/{num_samples}{'-'*20}")
        result = solver.solve(
            x0=np.random.uniform(
                low=class_list[key]().lb,
                high=class_list[key]().ub,
                size=class_list[key]().n
            )
        )
        print(result)
        results.append(result)
        print(f"{'-'*50}")

    print(os.path.join(os.path.dirname(__file__), "results", f"{problem.__class__.__name__}_pareto_front.png"))
    # sys.exit()

    # plotting pareto front
    plot_pareto_front(
        Y_true=problem.true_pareto_front,
        Y_approx=np.array([result['f'] for result in results]),
        problem_name=problem.__class__.__name__, # later append all of the settings to the string also
        save_path=os.path.join(os.path.dirname(__file__), "results", f"{problem.__class__.__name__}_pareto_front.png")
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
    metrics_res['problem'] = problem.__class__.__name__


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
