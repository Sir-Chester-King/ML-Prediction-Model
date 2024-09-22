"""
Machine Learning Model Project:
ML model developed to predict the potential purchase of a car by a person.
"""

# ---- IMPORT SECTION ---- #
from clear_console import *
from data_processing.encoding_categorical_data import *
from data_processing.handle_missing_data import handle_missing_data
from dataset import load_data


# ---- MAIN CODE SECTION ---- #
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

    # ---- CHECK FOR MISSING DATA ---- #
    if has_nan:
        variables_feature = handle_missing_data(variables_feature)
        print("-" * 40)
        print("-" * 40)
        print("New Dataset 'Features' without missing data", end='\n')
        print(variables_feature)
    # ---- CHECK FOR MISSING DATA ---- #

    # ---- HOT-ENCODING 'CATEGORICAL' DATA---- #

    # Hot-Coding to convert categorical data in <FEATURE> variables into values.
    variables_feature = hot_coding_FEATURE_variable(variables_feature)
    print("-" * 40)
    print("-" * 40)
    print("Hot-Encoded 'Features' Values", end='\n')
    print(variables_feature)

    # Hot-Coding to convert categorical data in <TARGET> variables into values.
    variables_target = hot_coding_TARGET_variables(variables_target)
    print("-" * 40)
    print("-" * 40)
    print("Hot-Encoded 'Target' Variables", end='\n')
    print(variables_target)


# ---- HOT-ENCODING 'CATEGORICAL' DATA---- #


if __name__ == '__main__':
    main()
