# ---- IMPORT SECTION ---- #
import os

import dotenv
import pandas

# ---- MAIN CODE SECTION ---- #

"""Machine learning model to predict the potential purchase of a car of a person, using 
the dataset as source input.

Import and processing DataSet.
"""


def main():
    dotenv.load_dotenv()  # Load the .env file to gather the path of CSV file.
    dataset = pandas.read_csv(os.getenv('path_data'))  # Gather the dataset from the CSV file.

    # Gather all the values in the columns from the dataset, except the last one; as a numpy array.
    # Used the ILOC indexer in PANDAS to select the subset of the data and store that in the variable.
    matrix_features_independent = dataset.iloc[:, :-1].values

    # Gather all the values in the last column from the dataset; as a numpy array.
    # Used the ILOC indexer in PANDAS to select the subset of the data and store that in the variable.
    vector_features_dependent = dataset.iloc[:, -1].values

    print("Matrix Features Independent:", end='\n')
    print(matrix_features_independent)

    print("Vector Features Dependent:", end='\n')
    print(vector_features_dependent)


# ---- MAIN CODE SECTION ---- #


# ---- TAKE CARE OF MISSING DATA SECTION ---- #
"""
Function to handle the missing values in the dataset.
To perform that, in the 'Salary' colum, in case of missing value, the value 
will be replaced with the average of all the values in the 'Salary' colum.
"""


def handle_missing_values():
    pass


# ---- TAKE CARE OF MISSING DATA SECTION ---- #


# __name__ is a special built-in variable that exists in every module (a module is simply a Python file).
# __main__ is a string that Python assigns to the __name__ variable when the module is executed as the main program.
# It serves as an entry point for the script execution.
# The if __name__ == "__main__": condition checks whether the script is being run directly or being imported.
# Code inside this if block will only execute if the script is run directly not when it is imported.
if __name__ == '__main__':
    main()
