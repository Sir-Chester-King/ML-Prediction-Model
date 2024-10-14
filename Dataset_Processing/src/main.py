# ---- IMPORT SECTION ---- #
from Dataset_Processing.dataset import load_data
from Dataset_Processing.encoding_categorical_data import *
from Dataset_Processing.handle_missing_data import handle_missing_data
from Dataset_Processing.scaling import *
from Dataset_Processing.split_data import *
from clear_console import *


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

    # ---- HOT-ENCODING 'CATEGORICAL' DATA ---- #

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
    # ---- HOT-ENCODING 'CATEGORICAL' DATA ---- #

    # ---- SPLIT TRAIN & TEST SET ---- #
    feature_train, feature_test, target_train, target_test = split_DATASET(variables_feature, variables_target)
    print("-" * 40)
    print("-" * 40)
    print("'Feature' TRAIN SET", end='\n')
    print(feature_train)

    print("-" * 40)
    print("-" * 40)
    print("'Feature' TEST SET", end='\n')
    print(feature_test)

    print("-" * 40)
    print("-" * 40)
    print("'Target' TRAIN SET", end='\n')
    print(target_train)

    print("-" * 40)
    print("-" * 40)
    print("'Target' TEST SET", end='\n')
    print(target_test)
    # ---- SPLIT TRAIN & TEST SET ---- #

    # ---- SCALING FEATURE ---- #
    feature_train, feature_test = scaling_FEATURE_variables(feature_train, feature_test)
    print("-" * 40)
    print("-" * 40)
    print("'Feature' TRAINING SCALED", end='\n')
    print(feature_train)

    print("-" * 40)
    print("-" * 40)
    print("'Feature' TEST SCALED", end='\n')
    print(feature_test)
    # ---- SCALING FEATURE ---- #


if __name__ == '__main__':
    main()
