# ---- IMPORT SECTION ---- #
import dotenv
import numpy
import pandas
from sklearn.impute import *

from clear_console import *

# ---- IMPORT SECTION ---- #

# ---- TAKE CARE OF MISSING DATA SECTION ---- #
"""
    Function to handle the missing values in the dataset.
    To perform that it be used the class SimpleImputer of SKLEARN library, used to handle missing data within a dataset.
    
    In case on missing data, them will be replaced with the average of all value in the colum.
    The data replaced are only numbers, not strings.
"""


def handle_missing_values(matrix_features_independent):
    """
        missing_values: Specifies the value to be considered missing.
        By default, it is set to numpy.nan), but can be set to other values if the missing data is
        represented differently.

        strategy: Defines the strategy for replacing missing values.
        mean option: Replaces missing values with the average of the column (only for numeric variables).
    """
    imputer_istance = SimpleImputer(missing_values=numpy.nan, strategy='mean')  # Object IMPUTER.

    """
        To handle the missing data in the dataset it used the .fit() function.
        The .fit() method of Imputer classes is used to calculate and store the statistics needed to 
        fill in missing data in a dataset. 
        
        In other words, fit() trains the Imputer on the data, identifying the information 
        needed to handle NaN values.
    """

    """
    CALCULATE THE STATISTICS NEEDED TO FILL IN MISSING DATA IN DATASET.
        Colum 1 = Age
        Colum 2 = Salary
        This links the Imputer instance to the matrix of features.
    """

    imputer_istance.fit(matrix_features_independent[:, 1:3])  # Specified all raw's and the first and second colum.

    """
    STORE THE PREVIOUSLY CALCULATED DATA IN THE MISSING DATA IN DATASET.
        Colum 1 = Age
        Colum 2 = Salary
    """
    # This method returns the new version od dataset (with the average of missing data).
    new_dataset = imputer_istance.transform(
        matrix_features_independent[:, 1:3])  # Passed as argument the columns of the dataset.

    """
    UPDATING THE NEW VERSION OF DATASET IN THE OLD DATASET, TO HAVE A DATASET WITHOUT MISSING DATA.
    """
    matrix_features_independent[:, 1:3] = new_dataset

    return matrix_features_independent
# ---- TAKE CARE OF MISSING DATA SECTION ---- #


# ---- LOAD DATA FROM CSV SECTION ---- #
def load_data():
    dotenv.load_dotenv()  # Load the .env file to gather the path of CSV file.
    dataset = pandas.read_csv(os.getenv('path_data'))  # Gather the dataset from the CSV file.

    # Check if there are missing values in the various columns (only numbers' value).
    has_nan = check_missing_value(dataset.iloc[:, 1:3].values)

    # Gather all the values in the columns from the dataset, except the last one; as a numpy array.
    # Used the ILOC indexer in PANDAS to select the subset of the data and store that in the variable.
    matrix_features_independent = dataset.iloc[:, :-1].values

    # Gather all the values in the last column from the dataset; as a numpy array.
    # Used the ILOC indexer in PANDAS to select the subset of the data and store that in the variable.
    vector_features_dependent = dataset.iloc[:, -1].values

    return matrix_features_independent, vector_features_dependent, has_nan


# ---- LOAD DATA FROM CSV SECTION ---- #


# ---- CHECK MISSING VALUE SECTION ---- #
# Check if there is/are missing values in the matrix_features_independent dataset.
def check_missing_value(matrix_features_independent):
    # Returns a boolean mask indicating the presence of NaN
    nan_mask = numpy.isnan(matrix_features_independent)

    # Returns True if there is at least one NaN value in the array, False otherwise.
    has_nan = numpy.any(nan_mask)

    if has_nan:
        return True


# ---- CHECK MISSING VALUE SECTION ---- #


# ---- MAIN CODE SECTION ---- #
"""Machine learning model to predict the potential purchase of a car of a person, using 
the dataset as source input.

Import and processing DataSet.
"""


def main():
    clear()
    matrix_features_independent, vector_features_dependent, has_nan = load_data()

    print("Matrix Features Independent:", end='\n')
    print(matrix_features_independent)

    print('\n')
    print("Vector Features Dependent:", end='\n')
    print(vector_features_dependent)

    if has_nan:
        print('\n')
        print("There is/are missing value", end='\n')
        matrix_features_independent = handle_missing_values(matrix_features_independent)

        print('\n')
        print("New Dataset without data missing", end='\n')
        print(matrix_features_independent)


# ---- MAIN CODE SECTION ---- #


# __name__ is a special built-in variable that exists in every module (a module is simply a Python file).
# __main__ is a string that Python assigns to the __name__ variable when the module is executed as the main program.
# It serves as an entry point for the script execution.
# The if __name__ == "__main__": condition checks whether the script is being run directly or being imported.
# Code inside this if block will only execute if the script is run directly not when it is imported.
if __name__ == '__main__':
    main()
