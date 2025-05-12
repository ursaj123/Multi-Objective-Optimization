import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../plotting"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
# print(os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../plotting")))
# sys.path.append('../problem_lists')
from problems import class_list
from pareto_front_plots import plot_2d_pareto_front, plot_3d_pareto_front

print("Importing done successfully")

for problem_name, problem_class in class_list.items():
    if problem_name=='CIRCULAR':
        continue
    
    print(f"Problem Name: {problem_name}")
    print(f"Problem Class: {problem_class}")

    # Create an instance of the problem
    problem_instance = problem_class()

    # Get the true Pareto front
    true_pareto_front = problem_instance.true_pareto_front

    # Print the true Pareto front
    print(f"True Pareto Front for {problem_name}:")
    print(true_pareto_front)

    # Plotting the true Pareto front
    if problem_instance.m == 2:
        plot_2d_pareto_front(Y_true=true_pareto_front, 
        problem_name=problem_name,
        save_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{problem_name}.png")
        )
    elif problem_instance.m == 3:
        plot_3d_pareto_front(f_true=true_pareto_front, 
        problem_name=problem_name,
        save_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{problem_name}.png")
        )




