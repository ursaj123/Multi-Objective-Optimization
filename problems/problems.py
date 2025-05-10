import importlib
import os
import sys
import numpy as np

def import_problems():
    folder_path = os.path.join(os.path.dirname(__file__), 'problem_lists')
    sys.path.append(folder_path)

    # List to hold (name, class) tuples
    class_list = {}

    # Iterate over Python files in the folder
    for file in os.listdir(folder_path):
        if file.endswith(".py") and not file.startswith("__"):
            module_name = file[:-3]  # Remove '.py'
            class_name = module_name.upper()  # Assuming class is capitalized version

            try:
                # Dynamically import module
                module = importlib.import_module(module_name)

                # Get the class from module
                cls_ = getattr(module, class_name)

                # Add to list
                class_list[class_name] = cls_
            except (ImportError, AttributeError) as e:
                print(f"Could not import {class_name} from {module_name}: {e}")

    # Done! You can now use class_list
    return class_list

class_list = import_problems()
# print(class_list)
# print(f"All problems are - {class_list.keys()}")
# print(class_list['JOS1']().evaluate_f(x = np.array([1., 2.])))




# Path to the folder

