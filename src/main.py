# ---- IMPORT SECTION ---- #
from clear_console import *
from data_processing.econding_category import *
from data_processing.handle_missing_data import handle_missing_data
from dataset import load_data

# ---- MAIN CODE SECTION ---- #
"""Machine learning model to predict the potential purchase of a car of a person, using 
the dataset as source input.
"""


def main():
    clear()
    variables_feature, variables_target, has_nan = load_data.load_data()
    clear()

    print("'Features' Variables:", end='\n')
    print(variables_feature)

    print("-" * 40)
    print("-" * 40)
    print("'Target' Variables:", end='\n')
    print(variables_target)

    if has_nan:
        variables_feature = handle_missing_data(variables_feature)

        print("-" * 40)
        print("-" * 40)
        print("New Dataset 'Features' without missing data", end='\n')
        print(variables_feature)

    # Hot-Coding to convert categorical dataset into values, to be manipulated for the machine learning model.
    variables_feature = hot_coding_variables_feature(variables_feature)

    print("-" * 40)
    print("-" * 40)
    print("Hot-Encoded 'Categorical Features' Values", end='\n')
    print(variables_feature)

    # Hot-Coding to convert categorical dataset into values, to be manipulated for the machine learning model.
    variables_target = hot_coding_variables_target(variables_target)

    print("-" * 40)
    print("-" * 40)
    print("Hot-Encoded 'Target' Variables", end='\n')
    print(variables_target)


# __name__ is a special built-in variable that exists in every module (a module is simply a Python file).
# __main__ is a string that Python assigns to the __name__ variable when the module is executed as the main program.
# It serves as an entry point for the script execution.
# The if __name__ == "__main__": condition checks whether the script is being run directly or being imported.
# Code inside this if block will only execute if the script is run directly not when it is imported.
if __name__ == '__main__':
    main()
